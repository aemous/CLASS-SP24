#include <immintrin.h>
#include <stdio.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is K-by-M, and B is K-by-N.
 * bi is the block-row of A. bj is the block-column of B, bk is the block number (of A and B).
 */
static void do_block(int lda, int ldaRounded, int M, int N, int K, int bi, int bj, int bk, double* A, double* B, double* C, double* dotProduct, double* AT, double* BBlock) {
//    double* AT = _mm_malloc(K * M * sizeof(double), 64); // K rows, M columns
    __m256d rowA1; // stores first quarter of row i of A
    __m256d rowA2; // stores second quarter of row i of A
    __m256d rowA3; // stores third quarter of row i of A
    __m256d rowA4; // stores fourth quarter of row i of A
    __m256d colB1; // stores first quarter of column j of B
    __m256d colB2; // stores second quarter of column j of B
    __m256d colB3; // stores third quarter of column j of B
    __m256d colB4; // stores fourth quarter of column j of B

    // Obtain a block-contiguous view on AT
    // For each column of AT-block
//    for (unsigned int j = 0; j < M; ++j) {
//        // For each row of AT-block
//        for (unsigned int i = 0; i < K; ++i) {
//            // Copy the element to AT
//            AT[i + j * BLOCK_SIZE] = A[i + j * ldaRounded];
//        }
//    }

    // transpose the A-block for SIMD-compatibility
    // For each column j of AT
//    for (unsigned int j = 0; j < M; ++j) {
//        // For each row i of AT
//        for (unsigned int i = 0; i < K; ++i) {
////            AT[i + j * K] = A[j + i * lda];
//            AT[i + j * BLOCK_SIZE] = A[j + i * lda];
//        }
//    }
//    for (unsigned int j = 0; j < M; ++j) {
        // For each row i of AT
//        for (unsigned int i = 0; i < K; ++i) {
//            AT[i + j * K] = A[j + bk + (i + bi) * ldaRounded];
//            if (i < K && j < M) {
//                AT[i + j * BLOCK_SIZE] = A[j + i * lda];
//            } else {
//                AT[i + j * BLOCK_SIZE] = 0;
//            }
//            if (i < K && j < N) {
//                BBLock[i + j * BLOCK_SIZE] = B[i + j * lda];
//            } else {
//                BBLock[i + j * BLOCK_SIZE] = 0;
//            }
//        }
//    }

    // this obtains a block-contiguous view on B. let's consider it later after we get this working
//    for (unsigned int j = 0; j < N; ++j) {
//        for (unsigned int i = 0; i < K; ++i) {
//            BBlock[i + j * K] = B[i + bk + (j + bj) * ldaRounded];
//        }
//    }

    // compute the multiplication
    // For each row i of A (col i of AT-block)
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
//            AAlignedPacked[i + j * K + bk * M * K + bi * M * ldaRounded]
//            rowA1 = _mm256_load_pd(A + i * K);
//            rowA2 = _mm256_load_pd(A + 4 + i * K);
//            rowA3 = _mm256_load_pd(A + 8 + i * K);
//            rowA4 = _mm256_load_pd(A + 12 + i * K);
//            colB1 = _mm256_load_pd(B + j * ldaRounded);
//            colB2 = _mm256_load_pd(B + 4 + j * ldaRounded);
//            colB3 = _mm256_load_pd(B + 8 + j * ldaRounded);
//            colB4 = _mm256_load_pd(B + 12 + j * ldaRounded);
            rowA1 = _mm256_load_pd(A + i * ldaRounded);
            rowA2 = _mm256_load_pd(A + 4 + i * ldaRounded);
            rowA3 = _mm256_load_pd(A + 8 + i * ldaRounded);
            rowA4 = _mm256_load_pd(A + 12 + i * ldaRounded);
            colB1 = _mm256_load_pd(B + j * ldaRounded);
            colB2 = _mm256_load_pd(B + 4 + j * ldaRounded);
            colB3 = _mm256_load_pd(B + 8 + j * ldaRounded);
            colB4 = _mm256_load_pd(B + 12 + j * ldaRounded);

            // compute first 'half' of the dot product of A[i,:] and B[:,j]
            __m256d dot1 = _mm256_hadd_pd(_mm256_mul_pd(rowA1, colB1), _mm256_mul_pd(rowA2, colB2));
//            __m256d dot1 = _mm256_mul_pd(rowA1, colB1);
            // compute second 'half' of the dot product of A[i,:] and B[:,j]
            __m256d dot2 = _mm256_hadd_pd(_mm256_mul_pd(rowA3, colB3), _mm256_mul_pd(rowA4, colB4));
//
//            // the sum of the 4 doubles in the vector below is the dot product of A[i,:] and B[:,j]
            _mm256_store_pd(dotProduct, _mm256_hadd_pd(dot1, dot2));
//            _mm256_store_pd(dotProduct, dot1);
            double cij = C[i + j * lda];
            for (int k = 0; k < 4; ++k) {
                cij += dotProduct[k];
            }
            C[i + j * lda] = cij;
        }
    }
//    free(AT);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    double* dotProduct = _mm_malloc(4 * sizeof(double), 64); // temp array of size 4 whose sum is a dot product
    double* AT = _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64); // temp array to store a transposed block
    double* BBlock = _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);

    // round lda up to the nearest BLOCK_SIZE multiple
    int ldaRounded = lda % BLOCK_SIZE == 0 ? lda : lda + (BLOCK_SIZE - (lda % BLOCK_SIZE));
    double* AAligned = _mm_malloc(ldaRounded * ldaRounded * sizeof(double), 32);
    double* BAligned = _mm_malloc(ldaRounded * ldaRounded * sizeof(double), 32);

    double* AAlignedPacked = _mm_malloc(ldaRounded * ldaRounded * sizeof(double), 64);

//    printf("LDA rounded: %d \n", ldaRounded);

//    printf("A: \n");
//    // For each row i of AT
//    for (unsigned int i = 0; i < lda; ++i) {
//        // For each column j of AT
//        for (unsigned int j = 0; j < lda; ++j) {
//            printf("%f ", A[i + j * lda]);
//        }
//        printf("\n");
//    }
//
//    printf("B: \n");
//    // For each row i of B
//    for (unsigned int i = 0; i < lda; ++i) {
//        // For each column j of AT
//        for (unsigned int j = 0; j < lda; ++j) {
//            printf("%f ", B[i + j * lda]);
//        }
//        printf("\n");
//    }


    // copy A to AAligned, pad with zeros
    for (unsigned int i = 0; i < ldaRounded; ++i) {
        for (unsigned int j = 0; j < ldaRounded; ++j) {
            AAligned[i + j * ldaRounded] = i < lda && j < lda ? A[j + i * lda] : 0;
        }
    }

    // copy B to BAligned, padded with zeros
    for (unsigned int i = 0; i < ldaRounded; ++i) {
        for (unsigned int j = 0; j < ldaRounded; ++j) {
            BAligned[i + j * ldaRounded] = i < lda && j < lda ? B[i + j * lda] : 0;
        }
    }

    // TODO one might consider doing the transpose and padding and packing all in one loop
    // Repack AAligned to AAlignedPacked
    // For each block-column of AAligned
    for (unsigned int bi = 0; bi < ldaRounded; bi += BLOCK_SIZE) {
        // For each block in this column
        for (unsigned int bk = 0; bk < ldaRounded; bk += BLOCK_SIZE) {
            // TODO theoretically we don't need min since padding guarantees it's a multiple of BLOCK_SIZE
            int M = min(BLOCK_SIZE, ldaRounded - bi);
            int K = min(BLOCK_SIZE, ldaRounded - bk);
            // For each column of the current block
            for (unsigned int j = 0; j < M; ++j) {
                // For each row of the current block
                for (unsigned int i = 0; i < K; ++i) {
//                    AAlignedPacked[i + j * K + bk * M * K + bi * M * ldaRounded] = AAligned[bk + i + (bi + j) * ldaRounded];
                    // sanity check: max value is BLOCK_SIZE-1 + (BLOCK_SIZE-1) * BLOCK_SIZE + (ldaRounded - BLOCK_SIZE) * BLOCK_SIZE * BLOCK_SIZE + (ldaRounded - BLOCK_SIZE) * BLOCK_SIZE * ldaRounded
                    // -1 + BLOCK_SIZE^2(1 - ldaRounded) + ldaRounded*BLOCK_SIZE^3 - BLOCK_SIZE^4 + ldaRounded^2*BLOCK_SIZE
                    // the sanity c heck too hard, lets just run
                }
            }
        }
    }

    // For each block-column of AAligned
    for (int bi = 0; bi < ldaRounded; bi += BLOCK_SIZE) {
        // For each block-column of BAligned
        for (int bj = 0; bj < ldaRounded; bj += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int bk = 0; bk < ldaRounded; bk += BLOCK_SIZE) {
//                printf("bi = %d, bj = %d, bk = %d", bi, bj ,bk);
                // Correct block dimensions if block "goes off edge of" the matrix
                // TODO theoretically we don't need min since padding guarantees it's a multiple of BLOCK_SIZE
                int M = min(BLOCK_SIZE, lda - bi);
                int N = min(BLOCK_SIZE, lda - bj);
                int K = min(BLOCK_SIZE, lda - bk);
//                int M = min(BLOCK_SIZE, ldaRounded - bi);
//                int N = min(BLOCK_SIZE, ldaRounded - bj);
//                int K = min(BLOCK_SIZE, ldaRounded - bk);
                // AAlignedPacked + bk * M * K + bi * M * ldaRounded
                // Perform individual block dgemm
                do_block(lda, ldaRounded, M, N, K, bi, bj, bk, AAligned + bk + bi * ldaRounded, BAligned + bk + bj * ldaRounded, C + bi + bj * lda, dotProduct, AT, BBlock);
//                do_block(lda, ldaRounded, M, N, K, bi, bj, bk, AAlignedPacked + bk * M * K + bi * M * ldaRounded, BAligned + bk + bj * ldaRounded, C + bi + bj * lda, dotProduct, AT, BBlock);
            }
        }
    }

//    printf("AAligned: \n");
//    // For each row i of AT
//    for (unsigned int i = 0; i < ldaRounded; ++i) {
//        // For each column j of AT
//        for (unsigned int j = 0; j < ldaRounded; ++j) {
//            printf("%f ", AAligned[i + j * ldaRounded]);
//        }
//        printf("\n");
//    }
//
//    printf("BAligned: \n");
//    // For each row i of BAligned
//    for (unsigned int i = 0; i < ldaRounded; ++i) {
//        // For each column j of BAligned
//        for (unsigned int j = 0; j < ldaRounded; ++j) {
//            printf("%f ", BAligned[i + j * ldaRounded]);
//        }
//        printf("\n");
//    }

//    // For each block-row of A
//    for (int i = 0; i < lda; i += BLOCK_SIZE) {
//        // For each block-column of B
//        for (int j = 0; j < lda; j += BLOCK_SIZE) {
//            // Accumulate block dgemms into block of C
//            for (int k = 0; k < lda; k += BLOCK_SIZE) {
////                printf("i = %d, j = %d, k = %d", i, j ,k);
//                // Correct block dimensions if block "goes off edge of" the matrix
//                // TODO theoretically we don't need min since padding guarantees it's a multiple of BLOCK_SIZE
//                int M = min(BLOCK_SIZE, lda - i);
//                int N = min(BLOCK_SIZE, lda - j);
//                int K = min(BLOCK_SIZE, lda - k);
//                // Perform individual block dgemm
//                do_block(lda, ldaRounded, M, N, K, i, j, k, AAligned + i + k * ldaRounded, BAligned + k + j * ldaRounded, C + i + j * lda, dotProduct, AT, BBlock);
//           }
//        }
//    }

    free(AT);
    free(dotProduct);
    free(BBlock);
    free(AAligned);
    free(AAlignedPacked);
    free(BAligned);
}
