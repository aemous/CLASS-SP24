const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <stdio.h>

// Define micro-block (kernel) size
#ifndef BI_SIZE
#define BI_SIZE 4
#endif

// TODO utilize this for higher-level block
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 41
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C_block := C_block + A_block * B_block
 *  where A, B, and C are lda-by-lda, and A_block is M-by-K, B_block is K-by-N, and C_Block is M-by-N
 */
static void do_micro_block(int lda,
                           int M,
                           int N,
                           int K,
                           unsigned int bj,
                           unsigned int bi,
                           unsigned int bk,
                           double* A,
                           double* B,
                           double* C,
                           double* A_block,
                           double* B_block,
                           double* C_block) {
    // Obtain a block-contiguous view on A transposed
    // TODO

    // Obtain a block-contiguous view on B
    // TODO

    // Obtain a block-contiguous view on C
    // TODO

    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/*
 * Assuming A and B are ldaxlda matrices, repack A into B. Repacks so that contiguous row-major blockSize elements are
 * adjacent in memory.
 */
void repack_row_major(int lda, int blockSize, double* A, double* B) {
    unsigned int idx = 0;
    // For each block row of A
    for (unsigned int bi = 0; bi < lda; bi += blockSize) {
        // For each block column of A
        for (unsigned int bk = 0; bk < lda; bk += blockSize) {
            // For each row of the block
            for (unsigned int i = 0; i < min(blockSize, lda - bi); ++i) {
                // For each column of the block
                for (unsigned int k = 0; k < min(blockSize, lda - bk); ++k) {
                    B[idx] = A[bi + i + (bk + k) * lda];
                    idx++;
                }
            }
        }
    }
}

/*
 * Assuming A and B are ldaxlda matrices, repack A into B. Repacks so that contiguous col-major blockSize elements are
 * adjacent in memory.
 */
void repack_col_major(int lda, int blockSize, double* A, double* B) {
    unsigned int idx = 0;
    // For each block column of A
    for (unsigned int bj = 0; bj < lda; bj += blockSize) {
        // For each block row of A
        for (unsigned int bk = 0; bk < lda; bk += blockSize) {
            // For each column of the block
            for (unsigned int j = 0; j < min(blockSize, lda - bj); ++j) {
                // For each row of the block
                for (unsigned int k = 0; k < min(blockSize, lda - bk); ++k) {
                    B[idx] = A[bk + k + (bj + j) * lda];
                    idx++;
                }
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    double* A_block = _mm_malloc(BI_SIZE * BI_SIZE * sizeof(double), 64);
    double* B_block = _mm_malloc(BI_SIZE * BI_SIZE * sizeof(double), 64);
    double* C_block = _mm_malloc(BI_SIZE * BI_SIZE * sizeof(double), 64);

//    double* A_repacked = _mm_malloc(lda * lda * sizeof(double), 64); // TODO experiment with 32 instead
    double A_repacked[lda * lda];
    double B_repacked[lda * lda];

    repack_row_major(lda, 4, A, A_repacked);
    repack_col_major(lda, 4, B, B_repacked);

    // TODO move these indices to the lowest level we can
    int B_idx = 0;
    int end_of_last_B_block = 0;
    int end_of_last_B_col = 0;


    // TODO implement a higher-level block size above the micro-level
    // For each block column bj of B
    for (unsigned int bj = 0; bj < lda; bj += BI_SIZE) {
        int A_idx = 0;
        int end_of_last_A_block = 0;
        // For each block row bi of A
        for (unsigned int bi = 0; bi < lda; bi += BI_SIZE) {
            // Accumulate block dgemms into block of C
            for (unsigned int bk = 0; bk < lda; bk += BI_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BI_SIZE, lda - bi);
                int N = min(BI_SIZE, lda - bj);
                int K = min(BI_SIZE, lda - bk);
                // todo worry about extracting to helper later
//                do_micro_block(lda, M, N, K, bj, bi, bk, A, B, C, A_block, B_block, C_block);

                // For each column j of B-block
                for (int j = 0; j < N; ++j) {
                    A_idx = end_of_last_A_block; // should be the start of the current A-block
                    // For each row i of A-block
                    for (int i = 0; i < M; ++i) {
                        B_idx = end_of_last_B_col;
                        // Compute C(i,j)
                        double cij = C[bi + i + (bj + j) * lda];

                        for (int k = 0; k < K; ++k) {
                            cij += A_repacked[A_idx] * B_repacked[B_idx];
//                            cij += A[i + k * lda] * B[k + j * lda];
                            A_idx++;
                            B_idx++;
                        }
                        C[bi + i + (bj + j) * lda] = cij;
                    }
                    end_of_last_B_col = B_idx;
                }
                // set end_of_last_A_block, since at this point we reached the end of prev A_block
                end_of_last_A_block = A_idx;
            }
        }
    }

//    A_block is M-by-K, B_block is K-by-N

    free(A_block);
    free(B_block);
    free(C_block);
//    free(A_repacked);

    // For each block-row of A
//    for (int i = 0; i < lda; i += BLOCK_SIZE) {
//        // For each block-column of B
//        for (int j = 0; j < lda; j += BLOCK_SIZE) {
//            // Accumulate block dgemms into block of C
//            for (int k = 0; k < lda; k += BLOCK_SIZE) {
//                // Correct block dimensions if block "goes off edge of" the matrix
//                int M = min(BLOCK_SIZE, lda - i);
//                int N = min(BLOCK_SIZE, lda - j);
//                int K = min(BLOCK_SIZE, lda - k);
//                // Perform individual block dgemm
//                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
//            }
//        }
//    }
}
