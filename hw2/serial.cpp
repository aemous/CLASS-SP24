#include "common.h"
#include <cmath>

#include <vector>

int num_cells = 0;
double cellSize = 0.;

std::vector<std::vector<std::vector<int>>> cells;

// TODO this may change if our cells end up not being square after parallelism
int get_cell_x(double size, double x) {
    return (int) ((num_cells - 1) * std::min((x / (size - cellSize)), 1.0));
}

// TODO this may change if our cells end up not being square after parallelism
int get_cell_y(double size, double y) {
    return (int) ((num_cells-1) * std::min((y / (size - cellSize)), 1.0));
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

    // TODO when we parallelize, the below might be a function of num processors/threads
    num_cells = floor(size / cutoff);
    cellSize = size / num_cells;
    int exp_parts_per_cell = ceil(1.0 * num_parts / num_cells);

    // initialize the grid of cells
    for (unsigned int i = 0; i < num_cells; ++i) {
        cells.push_back(std::vector<std::vector<int>>(num_cells));
        for (unsigned int j = 0; j < num_cells; ++j) {
            cells.at(i).push_back(std::vector<int>(exp_parts_per_cell + 10));
        }
    }

    // map the particles to their proper cells based on their position
    for (int p = 0; p < num_parts; ++p) {
        cells.at(get_cell_x(size, parts[p].x)).at(get_cell_y(size, parts[p].y)).push_back(p);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // next attempt:
    // for each particle in the world
        // get its cell
        // for each neighbor cell
            // compute force
    for (int i = 0; i < num_parts; ++i) {
        int cell_x = get_cell_x(size, parts[i].x);
        int cell_y = get_cell_y(size, parts[i].y);

        // TODO experiment with loop order?
        for (unsigned int jj = (int) fmax(0, cell_y-1); jj <= (int) fmin(num_cells-1, cell_y+1); ++jj) {
            for (unsigned int ii = (int) fmax(0, cell_x-1); ii <= (int) fmin(num_cells-1, cell_x+1); ++ii) {
                for (auto & neighbor_part : cells.at(ii).at(jj)) {
                    apply_force(parts[i], parts[neighbor_part]);
                }
            }
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
       move(parts[i], size);
    }

    // Clear cells
    for (int i = 0; i < num_cells; ++i) {
        for (int j = 0; j < num_cells; ++j) {
            cells.at(i).at(j).clear();
        }
    }

    // Recompute particle cells
    for (int i = 0; i < num_parts; ++i) {
        cells.at(get_cell_x(size, parts[i].x)).at(get_cell_y(size, parts[i].y)).push_back(i);
    }
}
