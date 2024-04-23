#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    std::vector<kmer_pair> data;
    std::vector<int> used;

    upcxx::global_ptr<kmer_pair> g_data;
    upcxx::global_ptr<uint64_t> g_used;

    upcxx::dist_object<upcxx::global_ptr<kmer_pair>> d_data;
    upcxx::dist_object<upcxx::global_ptr<uint64_t>> d_used;

    upcxx::atomic_domain<uint64_t> atomic_domain;

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);
    ~HashMap() {
        upcxx::delete_array(g_data);
        upcxx::delete_array(g_used);
    }

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    uint64_t get_target(const pkmer_t& kmer);

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    upcxx::future<bool> request_bin(upcxx::dist_object<upcxx::global_ptr<uint64_t>> d_used, uint64_t bin);
    bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) {
    my_size = size;
    data.resize(size);
    used.resize(size, 0);

    // initialize the atomic domain we'll use for reserving slots
    atomic_domain = upcxx::atomic_domain<uint64_t>({upcxx::atomic_op::compare_exchange});

    // allocate the global pointers
    g_data = upcxx::new_array<kmer_pair>(size);
    g_used = upcxx::new_array<uint64_t>(size);

    uint64_t *used_local = g_used.local();

    // initialize the g_used array with zeros
    for (unsigned int i = 0; i < size; ++i) {
        used_local[i] = 0;
    }

    // initialize the distributed objects
    d_data = upcxx::dist_object<upcxx::global_ptr<kmer_pair>>(g_data);
    d_used = upcxx::dist_object<upcxx::global_ptr<uint64_t>>(g_used);
}

bool HashMap::insert(const kmer_pair& kmer) {
    // get the target process
    uint64_t target_rank = get_target(kmer.kmer);

    // fetch the global pointers from those target processes
//    upcxx::global_ptr<kmer_pair> target_data = d_data.fetch(target_rank).wait();
//    upcxx::global_ptr<uint64_t> target_used = d_used.fetch(target_rank).wait();

    // linearly probe the slots, and atomically reserve the first empty one

    // write to the slot

    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();

        // TODO call request bin through an RPC on the distributed object d_used
        success = request_slot(slot);
        if (success) {
            // TODO call write through an RPC on the distributed d_data object
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
}

// TODO after getting correctness done, one might consider early-stopping at the first unused slot as an optimization
bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    // get the target process
    uint64_t target_rank = get_target(key_kmer);

    // TODO delete the next instruction

    // fetch the global pointers from those target processes
    upcxx::global_ptr<kmer_pair> target_data = d_data.fetch(target_rank).wait();
    upcxx::global_ptr<uint64_t> target_used = d_used.fetch(target_rank).wait();

    // linearly probe the bin, and atomically reserve the first empty one

    // set the value kmer at each iteration

    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            // TODO read from the bin using RPC on the distributed object
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

uint64_t HashMap::get_target(const pkmer_t& kmer) {
    return kmer.hash() % upcxx::rank_n();
}

bool HashMap::slot_used(uint64_t slot) { return used[slot] != 0; }

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
    kmer_pair *data_local = g_data.local();
    data_local[slot] = kmer;
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    kmer_pair *data_local = g_data.local();
    return data_local[slot];
}

bool HashMap::request_slot(uint64_t slot) {
    int dst = 0;
    atomic_domain.compare_exchange(g_used, g_used + slot, 0, &dst, std::memory_order_relaxed).wait();
    return dst != 0;
}

upcxx::future<bool> HashMap::request_bin(upcxx::dist_object<upcxx::global_ptr<uint64_t>> d_used, uint64_t bin) {
    return upcxx::rpc(get_target(kmer.kmer),
                      [](upcxx::dist_object<upcxx::global_ptr<uint64_t>> &bins, uint64_t bin, upcxx::atomic_domain<uint64_t> atomic_domain) -> bool {
                        int dst = 0;
                        atomic_domain.compare_exchange(g_used, g_used[slot], 0, &dst, std::memory_order_relaxed).wait();
                        return dst != 0;
                      }, d_used, bin, atomic_domain);
}

size_t HashMap::size() const noexcept { return my_size; }