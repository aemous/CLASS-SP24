#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int* parts_sorted;
int* bin_counts;
int num_cells;
thrust::device_ptr<int> bin_counts_ptr;
thrust::device_ptr<int> parts_sorted_ptr;

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

__global__ void compute_forces_gpu(particle_t* particles, int* parts_idx, int* prefix_sum, int num_parts, double size, int num_cells) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_x = (int)((particles[tid].x / size) * (num_cells-1));
    int bin_y = (int)((particles[tid].y / size) * (num_cells-1));

    particles[tid].ax = particles[tid].ay = 0;

    for (int y = bin_y - 1; y <= bin_y + 1; ++y) {
        for (int x = bin_x - 1; x <= bin_x + 1; ++x) {
            if (x >= 0 && x < num_cells && y >= 0 && y < num_cells){
                int bin_idx = x + y * num_cells;
                int pidx_start = bin_idx == 0 ? 0 : prefix_sum[bin_idx - 1];
                int pidx_end = prefix_sum[bin_idx];
                for (unsigned int k = pidx_start; k < pidx_end; ++k) {
                    int neighbor_idx = parts_idx[k];
                    apply_force_gpu(particles[tid], particles[neighbor_idx]);
                }
            }
        }
    }
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

__global__ void compute_bin_counts(particle_t* parts, int* b_counts, int num_parts, double size, int num_cells){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int cell_x = (int)((parts[tid].x / size) * (num_cells-1));
    int cell_y = (int)((parts[tid].y / size) * (num_cells-1));
    atomicAdd(&b_counts[cell_x + cell_y * num_cells], 1);
}

__global__ void compute_parts_sorted(particle_t* parts, int* parts_sorted, int* prefix_sum, int num_parts, double size, int num_cells){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_x = (int)((parts[tid].x / size) * (num_cells-1));
    int bin_y = (int)((parts[tid].y / size) * (num_cells-1));

    int binidx = bin_x + bin_y * num_cells;
    int parts_idx = binidx == 0 ? 0 : prefix_sum[binidx - 1];

    while (atomicCAS(&parts_sorted[parts_idx], -1, tid) != -1) {
        parts_idx++;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    num_cells = (int)(size/cutoff);
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    parts_sorted = new int[num_parts];
    cudaMalloc((void **)&parts_sorted, num_parts * sizeof(int));
    parts_sorted_ptr = thrust::device_pointer_cast(parts_sorted);

    cudaMalloc((void **)&bin_counts, (num_cells * num_cells) * sizeof(int));

    bin_counts_ptr = thrust::device_pointer_cast(bin_counts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    thrust::fill(bin_counts_ptr, bin_counts_ptr + (num_cells * num_cells), (int) 0);
    thrust::fill(parts_sorted_ptr, parts_sorted_ptr + num_parts, (int) -1);

    compute_bin_counts<<<blks, NUM_THREADS>>>(parts, bin_counts, num_parts, size, num_cells);

    thrust::inclusive_scan(bin_counts_ptr, bin_counts_ptr + (num_cells * num_cells), bin_counts_ptr);

    // compute parts copy sorted by bin id
    compute_parts_sorted<<<blks, NUM_THREADS>>>(parts, parts_sorted, bin_counts, num_parts, size, num_cells);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, parts_sorted, bin_counts, num_parts, size, num_cells);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
