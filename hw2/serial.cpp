#include "common.h"
#include <cmath>

#include <vector>
#include <iostream>

int num_cells = 0;
double cellSize = 0.;

std::vector<std::vector<std::vector<particle_t*>>> cells;

// TODO for more efficiency, we can map particle indices to their cell indices

// TODO this may change if our cells end up not being square after parallelism
int get_cell_x(double size, particle_t& p) {
    return (int) (num_cells * (p.x / size));
}

// TODO this may change if our cells end up not being square after parallelism
int get_cell_y(double size, particle_t& p) {
    return (int) (num_cells * (p.y / size));
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

    // TODO maybe add a check here, only update the cell for this particle if it left its cell ?
}


#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-emplace"
void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    std::cout << "Init called" << std::endl;

    // TODO when we parallelize, the below should be a function of num processors/threads
    num_cells = floor(size / 5*(2 * cutoff / size));
    cellSize = size / (num_cells-1);
    int exp_parts_per_cell = ceil(cellSize / size * num_parts);

//    std::cout << "Num cells " << num_cells << std::endl;
//    std::cout << "Cell size " << cellSize << std::endl;

//    std::cout << "Exp parts per cell " << exp_parts_per_cell << std::endl;

    // initialize the grid of cells
    for (unsigned int i = 0; i < num_cells; ++i) {
//        cells.push_back(std::vector<std::vector<particle_t*>>());
        cells.push_back(std::vector<std::vector<particle_t*>>(num_cells));
        for (unsigned int j = 0; j < num_cells; ++j) {
//            cells.at(i).push_back(std::vector<particle_t*>());
            cells.at(i).push_back(std::vector<particle_t*>(exp_parts_per_cell));
        }
    }

    // map the particles to their proper cells based on their position
    for (unsigned int p = 0; p < num_parts; ++p) {
        cells.at(get_cell_x(size, parts[p])).at(get_cell_y(size, parts[p])).push_back(&parts[p]);
    }
}
#pragma clang diagnostic pop

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

                int min_neighbor_i = (int) fmax(0, i-1);
                int max_neighbor_i = (int) fmin(num_cells-1, i+1);
                int min_neighbor_j = (int) fmax(0, j-1);
                int max_neighbor_j = (int) fmin(num_cells-1, j+1);

                for (unsigned int ii = min_neighbor_i; ii <= max_neighbor_i; ++ii) {
                    for (unsigned int jj = min_neighbor_j; jj <= max_neighbor_j; ++jj) {
                        for (unsigned int kk = 0; kk < cells.at(ii).at(jj).size(); ++kk) {
                            if (cells.at(ii).at(jj).at(kk)->x < row.at(j).at(k)->x
                            || cells.at(ii).at(jj).at(kk)->y < row.at(j).at(k)->y) {
                                // what if two different particles have identical positions ?
                                // they would be in the same cell with different k-values
                                // TODO if we fail accuracy, add this check
//                                std::cout << "Computing force between particles " << i << " " << j << " " << k << " " << ii << " " << jj << " " << kk << std::endl;
                                apply_force(*row.at(j).at(k), *cells.at(ii).at(jj).at(kk));
                            }
                        }
                    }
                }
            }
        }
    }

//    std::cout << "Applied all forces for this step." << std::endl;



//    std::cout << "Cleared all cells." << std::endl;

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);

        // remap the particle to their proper cells based on their position
//        cells.at(get_cell_x(size, parts[i])).at(get_cell_y(size, parts[i])).push_back(&parts[i]);
    }
//    std::cout << "All parts moved this step." << std::endl;
    for (unsigned int i = 0; i < num_cells; ++i) {
        std::vector<std::vector<particle_t *>> row = cells.at(i);
        for (unsigned int j = 0; j < num_cells; ++j) {
            // for all particles in the cell
            for (unsigned int k = 0; k < row.at(j).size(); ++k) {
                // if this particle's coords changed, swap its cell

                int cell_x = get_cell_x(size, *row.at(j).at(k));
                int cell_y = get_cell_y(size, *row.at(j).at(k));

                if (cell_x != i || cell_y != j) {
                    cells.at(cell_x).at(cell_y).push_back(row.at(j).at(k));
                    row.at(j).erase(row.at(j).begin() + k);
//                    std::cout << "Moved particle from " << i << " " << j << " to " << cell_x << " " << cell_y << std::endl;
                }
            }
        }
    }

}
#pragma clang diagnostic pop
