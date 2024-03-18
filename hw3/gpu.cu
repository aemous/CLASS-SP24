#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
thrust::device_vector<int> bin_counts;
thrust::device_vector<int> bin_end;
thrust::device_vector<int> sorted_particles;
int blks;
int num_cells = 0;

int get_cell_x(double size, double x) {
    return (int) ((num_cells-1) * x / size);
}

int get_cell_y(double size, double y) {
    return (int) ((num_cells-1) * y / size);
}

__device__ void apply_force_gpu(particle_t& particle, particle_t const &neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// in this function, we want to be able to access all of the bins in GPU memory, and each bin can be a different size.
__global__ void compute_forces_gpu(particle_t* particles, int* bin_counts, int* sorted_particles, int num_parts, int num_cells, int size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // zero the particle's acceleration
    particles[tid].ax = particles[tid].ay = 0;

    // compute the bin for the cell
    int cell_x = (int) ((num_cells-1) * particles[tid].x / size);
    int cell_y = (int) ((num_cells-1) * particles[tid].y / size);

    // for every neighbor cell
    for (unsigned int i = cell_x-1 > 0 ? cell_x-1 : 0; i < cell_x+1 < num_cells ? cell_x+1 : num_cells; ++i) {
        for (unsigned int j = cell_y-1 > 0 ? cell_y-1 : 0; j < cell_y+1 < num_cells ? cell_y+1 : num_cells; ++j) {
            int bin_idx = cell_x + cell_y*num_cells;
            // for every particle in the cell
            for (unsigned int k = bin_counts[bin_idx]; k < bin_idx == num_cells*num_cells-1 ? num_parts : bin_counts[bin_idx+1]; ++k) {
                int part_id = sorted_particles[k];
                apply_force_gpu(particles[tid], particles[part_id]);
            }
        }
    }

//    for (int j = 0; j < num_parts; j++)
//        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void compute_bin_counts_gpu(particle_t* particles, int* bin_counts, int num_parts, int num_cells, int size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int cell_x = (int) ((num_cells-1) * particles[tid].x / size);
    int cell_y = (int) ((num_cells-1) * particles[tid].y / size);
    int* addr = bin_counts + cell_x + cell_y*num_cells;
    int* rawAddr = &addr[0];
    atomicAdd(rawAddr, 1);
}

__global__ void compute_parts_sorted(particle_t* particles, int* parts_sorted, int* last_part, int* bin_counts, int num_parts, int num_cells, int size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // compute the bin i for the part
    int cell_x = (int) ((num_cells-1) * particles[tid].x / size);
    int cell_y = (int) ((num_cells-1) * particles[tid].y / size);

    // atomically increment last_part[i] (i.e. reserve an index of parts_sorted)
//    thrust::detail::normal_iterator<thrust::device_ptr<int>> addr = last_part.begin() + cell_x + cell_y*num_cells;
    int* addr = last_part + cell_x + cell_y*num_cells;
//    int* rawAddr = thrust::raw_pointer_cast(&addr[0]);
    int* rawAddr = &addr[0];
    int prev_last_part = atomicAdd(rawAddr, 1);
    // then, set parts_sorted[bin_counts[i] + last_part[i]] = part_id
    parts_sorted[bin_counts[cell_x + cell_y*num_cells] + prev_last_part + 1] = tid;
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS; // can we alter # of blocks ? ideally block map 1:1 with bins
    num_cells = floor(size / cutoff);

    bin_counts = thrust::device_vector<int>(num_cells * num_cells);
    bin_end = thrust::device_vector<int>(num_cells * num_cells);
    sorted_particles = thrust::device_vector<int>(num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // reset the vectors that support concurrent binning for computing via gpu
    thrust::fill(bin_counts.begin(), bin_counts.end(), 0);
    thrust::fill(bin_end.begin(), bin_end.end(), -1);

    // task: compute particle count per bin
    // for each particle (per gpu core)
        // compute the bin for the particle
        // increment the particle count for that bin using thrust::atomicAdd
    compute_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, thrust::raw_pointer_cast(bin_counts.data()), num_parts, num_cells, size);

    // print bin counts
//    for (const int& it : bin_counts) {
//        std::cout << "Count: " << it << std::endl;
//    }

    // task: prefix sum particle counts
    // use thrust::exclusive_scan on the particles/bin array. the last element should be num_parts
    thrust::exclusive_scan(bin_counts.begin(), bin_counts.end(), bin_counts.begin());

    // task: add the particle ids to a separate array parts_sorted
    // initialize an array, 1 entry for each cell, called last_part, initialized to -1
    // for each particle (per gpu core): then you compute
        // compute the bin i for the part
        // atomically increment last_part[i],
        // then, set parts_sorted[bin_counts[i] + last_part[i]] = part_id
//    compute_parts_sorted<<<blks, NUM_THREADS>>>(parts, sorted_particles, bin_end, bin_counts, num_parts, num_cells, size);

    // Compute forces
//    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, bin_counts, sorted_particles, num_parts, num_cells, size);

    // Move particles
//    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
