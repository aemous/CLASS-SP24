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
static void do_block(int lda, int M, int N, int K, int bi, int bj, int bk, double* A, double* B, double* C, double* dotProduct) {
    if (K < BLOCK_SIZE) {
        printf("Entered if\n");
        // K is not necessary a multiple of 4, so do not use SIMD
        // For each row i of A
        for (int i = 0; i < M; ++i) {
            // For each column j of B
            for (int j = 0; j < N; ++j) {
                double cij = C[i + j * lda];
                // For each element in the row/column
                for (int k = 0; k < K; ++k) {
                    cij += A[k + i * lda] * B[k + j * lda];
                }
                C[i + j * lda] = cij;
            }
        }
        return;
    }

    printf("Entered main body \n");

    __m256d rowA1; // stores first quarter of row i of A
    __m256d rowA2; // stores second quarter of row i of A
    __m256d rowA3; // stores third quarter of row i of A
    __m256d rowA4; // stores fourth quarter of row i of A
    __m256d colB1; // stores first quarter of column j of B
    __m256d colB2; // stores second quarter of column j of B
    __m256d colB3; // stores third quarter of column j of B
    __m256d colB4; // stores fourth quarter of column j of B

    // compute the multiplication
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            rowA1 = _mm256_load_pd(A + i * lda);
            rowA2 = _mm256_load_pd(A + 4 + i * lda);
            rowA3 = _mm256_load_pd(A + 8 + i * lda);
            rowA4 = _mm256_load_pd(A + 12 + i * lda);
            colB1 = _mm256_load_pd(B + j * lda);
            colB2 = _mm256_load_pd(B + 4 + j * lda);
            colB3 = _mm256_load_pd(B + 8 + j * lda);
            colB4 = _mm256_load_pd(B + 12 + j * lda);

            // compute first 'half' of the dot product of A[i,:] and B[:,j]
            __m256d dot1 = _mm256_hadd_pd(_mm256_mul_pd(rowA1, colB1), _mm256_mul_pd(rowA2, colB2));
            // compute second 'half' of the dot product of A[i,:] and B[:,j]
            __m256d dot2 = _mm256_hadd_pd(_mm256_mul_pd(rowA3, colB3), _mm256_mul_pd(rowA4, colB4));
//
//            // the sum of the 4 doubles in the vector below is the dot product of A[i,:] and B[:,j]
            _mm256_store_pd(dotProduct, _mm256_hadd_pd(dot1, dot2));
            double cij = C[i + j * lda];
            for (int k = 0; k < 4; ++k) {
                cij += dotProduct[k];
            }
            C[i + j * lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    double* dotProduct = _mm_malloc(4 * sizeof(double), 64); // temp array of size 4 whose sum is a dot product

    // round lda up to the nearest BLOCK_SIZE multiple
    double* AAligned = _mm_malloc(lda * lda * sizeof(double), 64);
    double* BAligned = _mm_malloc(lda * lda * sizeof(double), 64);

    // copy A to AAligned
    for (unsigned int i = 0; i < lda; ++i) {
        for (unsigned int j = 0; j < lda; ++j) {
            AAligned[i + j * lda] = A[j + i * lda];
        }
    }

    // copy B to BAligned, padded with zeros
    for (unsigned int i = 0; i < lda; ++i) {
        for (unsigned int j = 0; j < lda; ++j) {
            BAligned[i + j * lda] = B[i + j * lda];
        }
    }

    // For each block-column of AAligned
    for (int bi = 0; bi < lda; bi += BLOCK_SIZE) {
        // For each block-column of BAligned
        for (int bj = 0; bj < lda; bj += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int bk = 0; bk < lda; bk += BLOCK_SIZE) {
//                printf("bi = %d, bj = %d, bk = %d", bi, bj ,bk);
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - bi);
                int N = min(BLOCK_SIZE, lda - bj);
                int K = min(BLOCK_SIZE, lda - bk);
                // Perform individual block dgemm
                do_block(lda, M, N, K, bi, bj, bk, AAligned + bk + bi * lda, BAligned + bk + bj * lda, C + bi + bj * lda, dotProduct);
            }
        }
    }
    free(dotProduct);
    free(AAligned);
    free(BAligned);
}
