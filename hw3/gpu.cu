#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int num_cells = 0;

std::vector<std::vector<std::vector<particle_t>>> cells;

int get_cell_x(double size, double x) {
    return (int) ((num_cells-1) * x / size);
}

int get_cell_y(double size, double y) {
    return (int) ((num_cells-1) * y / size);
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

__global__ void compute_forces_gpu(particle_t* particles, thrust::device_vector<particle_t>* d_cells, int num_parts, int size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    int cell_x = get_cell_x(size, particles[tid].x);
    int cell_y = get_cell_y(size, particles[tid].y);

    for (unsigned int i = 0; i < d_cells[cell_x + num_cells*cell_y].size(); ++i) {
        apply_force_gpu(particles[tid], d_cells[cell_x + num_cells*cell_y][i]);
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

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS; // can we alter # of blocks ? ideally block map 1:1 with bins
    num_cells = floor(size / cutoff);

    // initialize the grid of cells
    for (unsigned int i = 0; i < num_cells; ++i) {
        cells.push_back(std::vector<std::vector<particle_t>>(num_cells));
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells.at(i).push_back(std::vector<particle_t>(ceil(1.0 * num_parts / num_cells) + 10));
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // entire plan (phase I):
        // store a 3D vector of particle copies on the CPU
        // Rebin particles entirely on the CPU.
        // Copy the bins to the GPU.
        // On the GPU, compute the force for the thread's particle, using the GPU-stored bin for the particle
        // On the GPU, move the thread's particle

    // Clear bins
    for (unsigned int i = 0; i < num_cells; ++i) {
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells[i][j].clear();
        }
    }

    // bin the particles
    for (int i = 0; i < num_parts; ++i) {
        int cell_x = get_cell_x(size, parts[i].x);
        int cell_y = get_cell_y(size, parts[i].y);
        for (unsigned int ii = std::max(0, cell_x-1); ii <= std::min(num_cells-1, cell_x+1); ++ii) {
            for (unsigned int jj = std::max(0, cell_y - 1); jj <= std::min(num_cells-1, cell_y + 1); ++jj) {
                cells.at(ii).at(jj).push_back(parts[i]);
            }
        }
    }

    // copy the cells to the GPU
    thrust::device_vector<particle_t> d_cells[num_cells*num_cells];
    for (unsigned int i = 0; i < num_cells; ++i) {
        for (unsigned int j = 0; j < num_cells; ++j) {
            d_cells[j + i*num_cells] = thrust::device_vector<particle_t>(
                    cells.at(i).at(j).begin(),
                    cells.at(i).at(j).end()
            );
        }
    }
    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, d_cells, num_parts, size);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
