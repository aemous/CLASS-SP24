#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int num_cells = 0;

// TODO after we achieve EC, one might consider thinking about warps

//std::vector<std::vector<std::vector<particle_t>>> cells;
thrust::device_vector<thrust::device_vector<particle_t>>> cells;
thrust::device_vector<thrust::device_vector<omp_lock_t>> grid_locks;

// TODO do we need these on the gpu? probably
__device__ int get_cell_x(double size, double x) {
    return (int) ((num_cells-1) * x / size);
}

__device__ int get_cell_y(double size, double y) {
    return (int) ((num_cells-1) * y / size);
}

__global__ void bin_particles_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int cell_x = get_cell_x(size, particles[tid].x);
    int cell_y = get_cell_y(size, particles[tid].y);

    omp_set_lock(&grid_locks.at(cell_x).at(cell_y));
    cells.at(cell_x).at(cell_y).insert(particles[tid]);
    omp_unset_lock(&grid_locks.at(cell_x).at(cell_y));
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
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

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    int cell_x = get_cell_x(size, particles[tid].x);
    int cell_y = get_cell_y(size, particles[tid].y);

    for (unsigned int ii = std::max(0, cell_x-1); ii <= std::min(num_cells-1, cell_x+1); ++ii) {
        for (unsigned int jj = std::max(0, cell_y - 1); jj <= std::min(num_cells-1, cell_y + 1); ++jj) {
            for (auto & kk : cells.at(ii).at(jj)) {
                apply_force_gpu(particles[tid], kk);
            }
        }
    }
//    for (int j = 0; j < num_parts; j++)
//        apply_force_gpu(particles[tid], particles[j]);
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

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    num_cells = floor(size / cutoff);

    for (unsigned int i = 0; i < num_cells; ++i) {
        cells.insert(thrust::device_vector<thrust::device_vector<particle_t>>(num_cells));
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells.at(i).insert(thrust::device_vector<particle_t>(ceil(1.0 * num_parts / num_cells) + 10));
        }
    }

    // initialize grid of locks
    for (unsigned int i = 0; i < num_cells; ++i) {
        grid_locks.insert(thrust::device_vector<omp_lock_t>(num_cells));
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // TODO one might consider parallelizing the clearing across gpu threads
    // Clear bins
    for (unsigned int i = 0; i < num_cells; ++i) {
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells.at(i).at(j).clear();
        }
    }

    // Rebin particles
    bin_particles_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
