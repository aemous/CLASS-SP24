#include "common.h"
#include <cmath>

#include <vector>
#include <unordered_set>

particle_t* parts;
int num_cells = 0;
double cellSize = 0.;

std::vector<std::vector<std::unordered_set<particle_t*>>> cells;

// TODO this may change if our cells end up not being square after parallelism
int get_cell_x(double size, particle_t& p) {
    return (int) ((num_cells - 1) * fmin((p.x / (size - cellSize)), 1.0));
}

// TODO this may change if our cells end up not being square after parallelism
int get_cell_y(double size, particle_t& p) {
    return (int) ((num_cells-1) * fmin((p.y / (size - cellSize)), 1.0));
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

void init_simulation(particle_t* inp_parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    parts = inp_parts;

    // TODO when we parallelize, the below should be a function of num processors/threads
    num_cells = floor(size / ((2 * cutoff / size)));
    cellSize = size / (num_cells-1);
    int exp_parts_per_cell = ceil(1.0 * num_parts / num_cells);

    // initialize the grid of cells
    for (unsigned int i = 0; i < num_cells; ++i) {
        cells.push_back(std::vector<std::unordered_set<particle_t*>>(num_cells));
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells.at(i).push_back(std::unordered_set<particle_t*>(exp_parts_per_cell));
        }
    }

    // map the particles to their proper cells based on their position
    for (int p = 0; p < num_parts; ++p) {
        cells.at(get_cell_x(size, parts[p])).at(get_cell_y(size, parts[p])).insert(&parts[p]);
    }
}

void simulate_one_step(particle_t* inp_parts, int num_parts, double size) {
    // next attempt:
    // for each particle in the world
        // get its cell
        // for each neighbor cell
            // compute force if particle id less than current particle id
    for (int i = 0; i < num_parts; ++i) {
        int cell_x = get_cell_x(size, parts[i]);
        int cell_y = get_cell_y(size, parts[i]);

        int min_neighbor_i = (int) fmax(0, cell_x-1);
        int max_neighbor_i = (int) fmin(num_cells-1, cell_x+1);
        int min_neighbor_j = (int) fmax(0, cell_y-1);
        int max_neighbor_j = (int) fmin(num_cells-1, cell_y+1);

        for (unsigned int ii = min_neighbor_i; ii <= max_neighbor_i; ++ii) {
            for (unsigned int jj = min_neighbor_j; jj <= max_neighbor_j; ++jj) {
                std::unordered_set<particle_t*> neighbor_cell = cells.at(ii).at(jj);
                for (auto & neighbor_part : neighbor_cell) {
                    // TODO if we fail accuracy, map particle pointers to their indices (or subclass particle type to add id?)
                    if (neighbor_part->x < parts[i].x) {
                        apply_force(parts[i], *neighbor_part);
                    }
                }
            }
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        int prev_cell_x = get_cell_x(size, parts[i]);
        int prev_cell_y = get_cell_y(size, parts[i]);

        move(parts[i], size);

        int cell_x = get_cell_x(size, parts[i]);
        int cell_y = get_cell_y(size, parts[i]);

        if (cell_x != prev_cell_x || prev_cell_y) {
            cells.at(cell_x).at(cell_y).insert(&parts[i]);
            cells.at(prev_cell_x).at(prev_cell_y).erase(&parts[i]);
        }
    }
}
