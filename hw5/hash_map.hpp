#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

static bool atomic_domain_initialized;

struct HashMap {
//    std::vector<kmer_pair> data;
//    std::vector<int> used;

    upcxx::global_ptr<kmer_pair> g_data;
    upcxx::global_ptr<uint64_t> g_used;

    upcxx::dist_object<upcxx::global_ptr<kmer_pair>> d_data;
    upcxx::dist_object<upcxx::global_ptr<uint64_t>> d_used;

    static upcxx::atomic_domain<uint64_t> ad;

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);
    ~HashMap() {
//        atomic_domain.destroy();
        upcxx::delete_array(g_data);
        upcxx::delete_array(g_used);
    }

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    uint64_t get_target(const pkmer_t& kmer);

//    static upcxx::atomic_domain<uint64_t> get_atomic_domain() {
//        if (!atomic_domain_initialized) {
//            HashMap::ad = upcxx::atomic_domain<uint64_t>({upcxx::atomic_op::compare_exchange});
//            atomic_domain_initialized = true;
//        }
//        return HashMap::ad;
//    }

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
//    bool request_slot(uint64_t slot);
};

HashMap::HashMap(size_t size) {
//    init_atomic_domain();
    my_size = size;
//    data.resize(size);
//    used.resize(size, 0);

    // initialize the atomic domain we'll use for reserving slots
//    atomic_domain = upcxx::atomic_domain<uint64_t>({upcxx::atomic_op::compare_exchange});
//    HashMap::ad = upcxx::atomic_domain<uint64_t>({upcxx::atomic_op::compare_exchange});

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
    std::cout << "Begin insert" << std::endl;
    uint64_t target_rank = get_target(kmer.kmer);

    // this rpc should do everything
    // TODO i have suspicions that this capture clause might not do what i want
    // will the instance fields referenced below reference the fields on the remote process or the caller ?
    // we want it to be the remote process. if it's the caller, we SHOULD get an error when calling .local() on the global ptrs
    std::cout << "About to define future " << std::endl;
    upcxx::future<bool> future = upcxx::rpc(target_rank,
                                            [](const kmer_pair& kmer, const size_t size) -> bool {
                                                std::cout << "Begin RPC" << std::endl;
                                                uint64_t hash = kmer.hash();
                                                uint64_t probe = 0;
                                                bool success = false;

                                                std::cout << "About to enter do-while" << std::endl;

                                                do {
                                                    std::cout << "Begin do-while, size: " << size << std::endl;
                                                    uint64_t bin = (hash + probe++) % size;

                                                    std::cout << "Bin " << unsigned(bin) << std::endl;

                                                    // attempt to request the bin
                                                    uint64_t* used_local = g_used.local();
                                                    std::cout << "Call to local succes" << std::endl;
                                                    uint64_t result = HashMap::ad.compare_exchange(g_used + bin, (uint64_t) 0, (uint64_t) 1, std::memory_order_relaxed).wait();
                                                    std::cout << "Call to compare exchange succ" << std::endl;
//                                                    success = used_local[bin] != 0;
                                                    std::cout << "Success = " << unsigned(result) << std::endl;
                                                    success = result == 0;
                                                    if (success) {
                                                        // write to the bin
                                                        kmer_pair *data_local = g_data.local();
                                                        data_local[bin] = kmer;
                                                    }
                                                } while (!success && probe < size);

                                                return success;
                                            }, kmer, size());
    return future.wait();
}

// TODO after getting correctness done, one might consider early-stopping at the first unused slot as an optimization
bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    // get the target process
    uint64_t target_rank = get_target(key_kmer);

    upcxx::future<kmer_pair> future = upcxx::rpc(target_rank,
                                            [this](const pkmer_t key_kmer) -> kmer_pair {
                                                uint64_t hash = key_kmer.hash();
                                                uint64_t probe = 0;
                                                bool success = false;
                                                kmer_pair output = kmer_pair();

                                                do {
                                                    uint64_t bin = (hash + probe++) % size();

                                                    uint64_t* used_local = g_used.local();
                                                    kmer_pair* data_local = g_data.local();

                                                    if (used_local[bin] != 0) {
                                                        output = data_local[bin];
                                                        if (output.kmer == key_kmer) {
                                                            success = true;
                                                        }
                                                    }
                                                } while (!success && probe < size());
                                                // if success is false, return the 'error kmer'

                                                if (!success) {
                                                    output.fb_ext[0] = 'N';
                                                    output.fb_ext[1] = 'N';
                                                }

                                                return output;
                                            }, key_kmer);
    // set val_kmer to the output of the rpc future, check if we received the 'error kmer', and return false only if we did
    val_kmer = future.wait();

    if (val_kmer.fb_ext[0] == 'N' && val_kmer.fb_ext[1] == 'N') {
        return false;
    }

    return true;
}

uint64_t HashMap::get_target(const pkmer_t& kmer) {
    return kmer.hash() % upcxx::rank_n();
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
    kmer_pair *data_local = g_data.local();
    data_local[slot] = kmer;
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    kmer_pair *data_local = g_data.local();
    return data_local[slot];
}

//bool HashMap::request_slot(uint64_t slot) {
//    int dst = 0;
//    atomic_domain.compare_exchange(g_used, g_used + slot, 0, &dst, std::memory_order_relaxed).wait();
//    return dst != 0;
//}

size_t HashMap::size() const noexcept { return my_size; }