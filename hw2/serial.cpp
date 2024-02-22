#include "common.h"
#include <cmath>

#include <vector>

int num_cells = 0;
double cellSize = 0.;

std::vector<std::vector<std::vector<particle_t*>>> cells;

// TODO for more efficiency, we can map particle indices to their cell indices

// TODO this may change if our cells end up not being square after parallelism
static int get_cell_x(const int num_cells, double size, particle_t& p) {
    return (int) (num_cells * p.x / size);
}

// TODO this may change if our cells end up not being square after parallelism
static int get_cell_y(const int num_cells, double size, particle_t& p) {
    return (int) (num_cells * p.y / size);
}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    // TODO when we parallelize, the below should be a function of num processors/threads
    int num_cells = floor(size / (2 * cutoff / size));
    double cellSize = size / num_cells-1;
    int exp_parts_per_cell = ceil(cellSize / size * num_parts);

    // initialize the grid of cells
    for (unsigned int i = 0; i < num_cells; ++i) {
        cells.at(i) = std::vector<std::vector<particle_t*>>(num_cells);
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells.at(i).at(j) = std::vector<particle_t*>(exp_parts_per_cell);
        }
    }

    // map the particles to their proper cells based on their position
    for (unsigned int p = 0; p < num_parts; ++p) {
        cells.at(get_cell_x(num_cells, size, parts[p])).at(get_cell_y(num_cells, size, parts[p])).push_back(&parts[p]);
    }
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-loop-convert"
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // TODO this would be a good place to, for example, compute forces between particles
    // and other particles in the relevant cells of the screen tree.

    for (unsigned int i = 0; i < num_cells; ++i) {
        std::vector<std::vector<particle_t*>> row = cells.at(i);
        for (unsigned int j = 0; j < num_cells; ++j) {
            for (unsigned int k = 0; k < row.at(j).size(); ++k){
                // compute force between particle k and all other relevant particles left or above k
                row.at(j).at(k)->ax = row.at(j).at(k)->ay = 0;

                int min_neighbor_i = (int) fmin(0, i-1);
                int max_neighbor_i = (int) fmax(num_cells-1, i+1);
                int min_neighbor_j = (int) fmin(0, j-1);
                int max_neighbor_j = (int) fmax(num_cells-1, j+1);

                for (unsigned int ii = min_neighbor_i; ii <= max_neighbor_i; ++ii) {
                    for (unsigned int jj = min_neighbor_j; jj <= max_neighbor_j; ++jj) {
                        for (unsigned int kk = 0; kk < cells.at(ii).at(jj).size(); ++kk) {
                            if (cells.at(ii).at(jj).at(kk)->x < row.at(j).at(k)->x
                            || cells.at(ii).at(jj).at(kk)->y < row.at(j).at(k)->y) {
                                // what if two different particles have identical positions ?
                                // they would be in the same cell with different k-values
                                // TODO if we fail accuracy, add this check
                                apply_force(*row.at(j).at(k), *cells.at(ii).at(jj).at(kk));
                            }
                        }
                    }
                }
            }
        }
    }

    // clear each cell
    for (unsigned int i = 0; i < num_cells; ++i) {
        std::vector<std::vector<particle_t *>> row = cells.at(i);
        for (unsigned int j = 0; j < num_cells; ++j) {
            row.at(j).clear();
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);

        // remap the particle to their proper cells based on their position
        cells.at(get_cell_x(num_cells, size, parts[i])).at(get_cell_y(num_cells, size, parts[i])).push_back(&parts[i]);
    }
}
#pragma clang diagnostic pop
