#include <stdio.h>
#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 36
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * So, it looks like this implementation basically does the naive dgemm, except on smaller
 * blocks instead of the whole matrix.
 */

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
 * So, it looks like this implementation basically does the naive dgemm, except on smaller
 * blocks instead of the whole matrix.
 */

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is K-by-M, and B is K-by-N.
 */
static void do_block_row_major_A(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each column i of transposed A (i.e. row of A)
    for (unsigned int i = 0; i < M; ++i) {
        // For each column j of B
        for (unsigned int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (unsigned int k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

// TODO technically below code works since square, but its reading incorrectly, fix later
/*
 * Transposes matrix A of size MxM in-place.
 */
void transpose(double* A, int M) {
    // we access a column-major matrix as:
    // A[i,j] = A[row + col * total_rows]
    // For each row i of A
    double temp = 0;
    for (unsigned int i = 0; i < M - 1; ++i) {
        // For each column j of A
        for (unsigned int j = i + 1; j < M; ++j) {
            if (i == j) {
                // diagonal elements need not be touched in a transpose operation
                continue;
            }
            temp = A[i + j * M];
            A[i + j * M] = A[j + i * M];
            A[j + i * M] = temp;
        }
    }
}

// TODO: this will probaably become obsolete once we do m_alloc (or maybe not?) who cares
// TODO: technically code below works because we only call it with M=N, but it reads incorrectly, fix it later
/*
 * Given a M x N column-major matrix A, copy its contents into matrix B column-major
 */
void deep_copy(int M, int N, double* A, double* B) {
    // For each column i of A
    for (unsigned int i = 0; i < N; ++i) {
        // For each row j of A
        for (unsigned int j = 0; j < M; ++j) {
            // Copy A[j,i] into B[j,i]
            B[j + i * N] = A[j + i * N];
            // if i did j + i * M, max would be: (M-1) + (N-1) * M = M - 1 + NM - m
            // max value of index: (M-1) + (N-1) * N = M - 1 + N^2 - N
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format
 * On exit, A and B maintain their input values. */
void square_dgemm_row_major_A(int lda, double* A, double* B, double* C) {
    // here, we'll transpose A, and after that it'll still be lda x lda

//    printf("Input A: %f %f %f %f \n %f %f %f %f \n %f %f %f %f \n %f %f %f %f", A_dont_touch[0], A_dont_touch[4], A_dont_touch[8], A_dont_touch[12], A_dont_touch[1], A_dont_touch[5], A_dont_touch[9], A_dont_touch[13], A_dont_touch[2], A_dont_touch[6], A_dont_touch[10], A_dont_touch[14], A_dont_touch[3], A_dont_touch[7], A_dont_touch[11], A_dont_touch[15]);
//    double* A = malloc(lda * lda * sizeof(double));
//    deep_copy(lda, lda, A_dont_touch, A);
//    printf("Input A copied: %f %f %f %f \n %f %f %f %f \n %f %f %f %f \n %f %f %f %f", A[0], A[4], A[8], A[12], A[1], A[5], A[9], A[13], A[2], A[6], A[10], A[14], A[3], A[7], A[11], A[15]);
//    printf("Input B: %f %f %f %f \n %f %f %f %f \n %f %f %f %f \n %f %f %f %f", B[0], B[4], B[8], B[12], B[1], B[5], B[9], B[13], B[2], B[6], B[10], B[14], B[3], B[7], B[11], B[15]);
//    printf("Input C: %f %f %f %f \n %f %f %f %f \n %f %f %f %f \n %f %f %f %f", C[0], C[4], C[8], C[12], C[1], C[5], C[9], C[13], C[2], C[6], C[10], C[14], C[3], C[7], C[11], C[15]);
    transpose(A, lda);
//    printf("Transposed A to: %f %f %f %f \n %f %f %f %f \n %f %f %f %f \n %f %f %f %f", A[0], A[4], A[8], A[12], A[1], A[5], A[9], A[13], A[2], A[6], A[10], A[14], A[3], A[7], A[11], A[15]);

    // TODO i verified transpose works. but something is wrong with the code below (possibly including do_block_row_major_A

    // the big question: does it matter how we iterate the blocks ? I think so.
    // I think it'll be more efficient to iterate the blocks in the direction of the 'majorness' of the matrix.

    // next big question: do we need to adjust m, n, k? i dont think so.

    // then why was M unused in do_block_row_major_A ? because i was dumb

    // For each block-column of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i); // shouldnt this be lda - i - 1 ? same for below
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm

                // the 5th argument in the call below is the top-left entry of the block.
                // is this correct then ?

                // col major matrix :
                // A[i,j] = A[row + col * total_columns]
                do_block_row_major_A(lda, M, N, K, A + k + i * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }

//    printf("Result C: %f %f %f %f \n %f %f %f %f \n %f %f %f %f \n %f %f %f %f", C[0], C[4], C[8], C[12], C[1], C[5], C[9], C[13], C[2], C[6], C[10], C[14], C[3], C[7], C[11], C[15]);
//    free(A);
    transpose(A, lda);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // here, we'll transpose A, and after that it'll still be lda x lda

    square_dgemm_row_major_A(lda, A, B, C);

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
