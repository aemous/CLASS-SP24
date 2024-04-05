#include "common.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

// Put any static global variables here that you will use throughout the simulation.
int num_bins;
std::vector<std::vector<particle_t>> initial_bins;

std::vector<std::vector<particle_t*>> local_bins;
std::vector<particle_t> local_first_bin;
std::vector<particle_t> local_last_bin;
std::vector<particle_t> finer_bins;

std::vector<particle_t> redis_up;
std::vector<particle_t> redis_down;

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

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    int target_bin_size = (int)(size/cutoff);
    num_bins = ((target_bin_size / num_procs) * num_procs);
    local_bins = std::vector<std::vector<particle_t *>>(num_bins/num_procs);
    initial_bins = std::vector<std::vector<particle_t>>(num_procs);
    MPI_Status status;
    MPI_Status status2;

    if (rank == 0){
        // perform an initial per-process binning for distribution
        for (unsigned int i = 0; i < num_parts; ++i){
            int bin_y = (int) ((parts[i].y / size) * num_procs);
            initial_bins[bin_y].push_back(parts[i]);
        }
        for (unsigned int i = 1; i < num_procs; ++i){
            int size = initial_bins[i].size();
            MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(initial_bins[i].data(), size, PARTICLE, i, 0, MPI_COMM_WORLD);
        }
        finer_bins = initial_bins[0];
    } else {
        int incoming_num_parts = 0;
        MPI_Recv(&incoming_num_parts, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status2);
        finer_bins = std::vector<particle_t>(incoming_num_parts);
        MPI_Recv(&finer_bins[0], incoming_num_parts, PARTICLE, 0, 0, MPI_COMM_WORLD, &status);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // clear all bins to be updated this timestep
    local_first_bin.clear();
    local_last_bin.clear();
    redis_up.clear();
    redis_down.clear();

    for (unsigned int i = 0; i < local_bins.size(); ++i){
        local_bins[i].clear();
    }

    for (unsigned int i = 0; i < finer_bins.size(); ++i){
        double sub_size = size / num_procs;
        double frac = (finer_bins[i].y - (rank * sub_size))/sub_size;
        int num_proc_bins = local_bins.size();
        int bin_y = (int)(frac * num_proc_bins);
        if(frac < 1 && frac >= 0){
            local_bins[bin_y].push_back(&finer_bins[i]);
        }
    }

    if(rank != 0) {
        std::vector<particle_t> objects;
        for(int i = 0; i < local_bins.size(); ++i){
            objects.push_back(*local_bins[0][i]);
        }
        int top_size_s = objects.size();
        int top_size_r;
        MPI_Sendrecv(&top_size_s, 1, MPI_INT, rank-1, 0, &top_size_r, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (top_size_s > 0 && top_size_r > 0){
            std::vector<particle_t> temp(top_size_r);
            MPI_Sendrecv(&objects[0], top_size_s, PARTICLE, rank-1, 0, &temp[0], top_size_r, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < top_size_r; ++i){
                local_first_bin.push_back(temp[i]);
            }
        }
    }

    if (rank != num_procs - 1){
        std::vector<particle_t> objects;
        for (int i = 0; i < local_bins[local_bins.size() - 1].size(); ++i) {
            objects.push_back(*local_bins[local_bins.size() - 1][i]);
        }
        int bot_size_s = objects.size();
        int bot_size_r;
        MPI_Sendrecv(&bot_size_s, 1, MPI_INT, rank+1, 0, &bot_size_r, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (bot_size_s > 0 && bot_size_r > 0){
            std::vector<particle_t> temp(bot_size_r);
            MPI_Sendrecv(&objects[0], bot_size_s, PARTICLE, rank+1, 0, &temp[0], bot_size_r, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < bot_size_r; i++) {
                local_last_bin.push_back(temp[i]);
            }
        }
    }

    for (unsigned int j = 0; j < local_bins.size(); ++j){
        for (unsigned int i = 0; i < local_bins[j].size(); ++i) {
            local_bins[j][i]->ax = (local_bins[j][i]->ay = 0);
            for (int y = j - 1; y <= j + 1; ++y) {
                if (y == -1) {
                    for (unsigned int l = 0; l < local_first_bin.size(); ++l) {
                        apply_force(*local_bins[j][i], local_first_bin[l]);
                    }
                } else if (y == local_bins.size()){
                    for (int l = 0; l < local_last_bin.size(); ++l) {
                        apply_force(*local_bins[j][i], local_last_bin[l]);
                    }
                } else if (y >= 0 && y < local_bins.size()) {
                    for (unsigned int l = 0; l < local_bins[y].size(); ++l) {
                        apply_force(*local_bins[j][i], *local_bins[y][l]);
                    }
                }
            }
        }
    }

    for (unsigned int i = 0; i < finer_bins.size(); ++i) {
        move(finer_bins[i], size);
    }

    for (int i = finer_bins.size() - 1; i >= 0; --i){
        double sub_size = size / num_procs;
        double frac = (finer_bins[i].y - (rank * sub_size))/sub_size;
        int num_proc_bins = local_bins.size();
        int bin_y = (int)(frac * num_proc_bins);

        if (frac >= 1) {
            redis_down.push_back(finer_bins[i]);
            finer_bins.erase(finer_bins.begin() + i);
        } else if (frac < 0) {
            redis_up.push_back(finer_bins[i]);
            finer_bins.erase(finer_bins.begin() + i);
        }
    }

    if (rank != 0) {
        int top_size_s = redis_up.size();
        int top_size_r;
        MPI_Sendrecv(&top_size_s, 1, MPI_INT, rank-1, 0, &top_size_r, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(top_size_s > 0 || top_size_r > 0){
            std::vector<particle_t> temp(top_size_r);
            MPI_Sendrecv(&redis_up[0], top_size_s, PARTICLE, rank-1, 0, &temp[0], top_size_r, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int i = 0; i < top_size_r; i++){
                finer_bins.push_back(temp[i]);
            }
        }
    }

    if(rank != num_procs - 1){
        int bot_size_s = redis_down.size();
        int bot_size_r;
        MPI_Sendrecv(&bot_size_s, 1, MPI_INT, rank+1, 0, &bot_size_r, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (bot_size_s > 0 || bot_size_r > 0) {
            std::vector<particle_t> temp(bot_size_r);
            MPI_Sendrecv(&redis_down[0], bot_size_s, PARTICLE, rank+1, 0, &temp[0], bot_size_r, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < bot_size_r; i++){
                finer_bins.push_back(temp[i]);
            }
        }
    }

}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    std::vector<particle_t> parts_copy;
    if (rank == 0){
        parts_copy = std::vector<particle_t>(num_parts);
    }

    int *rcounts, *displs;
    rcounts = (int*) malloc(num_procs*sizeof(int));
    int size_to_gather = finer_bins.size();
    MPI_Gather(&size_to_gather, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    displs = (int*) malloc(num_procs*sizeof(int));
    displs[0] = 0;
    for (unsigned int i = 1; i < num_procs; ++i) {
        displs[i] = displs[i-1] + rcounts[i-1];
    }

    MPI_Gatherv(&finer_bins[0], finer_bins.size(), PARTICLE, &parts_copy[0], rcounts, displs, PARTICLE, 0, MPI_COMM_WORLD);
    if (rank == 0){
        for (unsigned int i = 0; i < num_parts; ++i){
            parts[parts_copy[i].id - 1].x = parts_copy[i].x;
            parts[parts_copy[i].id - 1].y = parts_copy[i].y;
        }
    }

    free(rcounts);
    free(displs);
}