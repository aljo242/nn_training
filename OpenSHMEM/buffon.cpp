#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <shmem.h>

static const double PI = 3.141592653589793238462643;

int
buffon_laplace_simulate(double a, double b, double l, int trial_num)
{
    double angle;
    int hits;
    int trial;
    double x1;
    double x2;
    double y1;
    double y2;

    hits = 0;

    for (trial = 1; trial <= trial_num; ++trial) {
        //
        // Randomly choose the location of the eye of the needle in
        // [0,0]x[A,B],
        // and the angle the needle makes.
        //
        x1 = a * (double) rand() / (double) RAND_MAX;
        y1 = b * (double) rand() / (double) RAND_MAX;
        angle = 2.0 * PI * (double) rand() / (double) RAND_MAX;
        //
        // Compute the location of the point of the needle.
        //
        x2 = x1 + l * cos(angle);
        y2 = y1 + l * sin(angle);
        //
        // Count the end locations that lie outside the cell.
        //
        if (x2 <= 0.0 || a <= x2 || y2 <= 0.0 || b <= y2) {
            ++hits;
        }
    }
    return hits;
}

double
r8_abs(double x)
{
    return (0.0 <= x) ? x : (-x);
}

double
r8_huge()
{
    return 1.0E+30;
}

//
// symmetric variables for reduction
//
int pWrk[SHMEM_REDUCE_SYNC_SIZE];
long pSync[SHMEM_BCAST_SYNC_SIZE];

int hit_total;
int hit_num;

int
main()
{
    const double a = 1.0;
    const double b = 1.0;
    const double l = 1.0;
    const int master = 0;
    double pdf_estimate;
    double pi_error;
    double pi_estimate;
    int process_num;
    int process_rank;
    double random_value;
    int seed;
    int trial_num = 100000;
    int trial_total;

    //
    // Initialize SHMEM.
    //
    shmem_init();
    //
    // Get the number of processes.
    //
    process_num = shmem_n_pes();
    //
    // Get the rank of this process.
    //
    process_rank = shmem_my_pe();
    //
    // The master process prints a message.
    //
    if (process_rank == master) {
        std::cout << "\n";
        std::cout << "BUFFON_LAPLACE - Master process:\n";
        std::cout << "  C++ version\n";
        std::cout << "\n";
        std::cout << "  A SHMEM example program to estimate PI\n";
        std::cout << "  using the Buffon-Laplace needle experiment.\n";
        std::cout << "  On a grid of cells of  width A and height B,\n";
        std::cout << "  a needle of length L is dropped at random.\n";
        std::cout << "  We count the number of times it crosses\n";
        std::cout << "  at least one grid line, and use this to estimate \n";
        std::cout << "  the value of PI.\n";
        std::cout << "\n";
        std::cout << "  The number of processes is " << process_num << "\n";
        std::cout << "\n";
        std::cout << "  Cell width A =    " << a << "\n";
        std::cout << "  Cell height B =   " << b << "\n";
        std::cout << "  Needle length L = " << l << "\n";
    }
    //
    // added barrier here to force output sequence
    //
    shmem_barrier_all();
    //
    // Each process sets a random number seed.
    //
    seed = 123456789 + process_rank * 100;
    srand(seed);
    //
    // Just to make sure that we're all doing different things, have each
    // process print out its rank, seed value, and a first test random value.
    //
    random_value = (double) rand() / (double) RAND_MAX;

    std::cout << "  " << std::setw(8) << process_rank
     << "  " << std::setw(12) << seed
     << "  " << std::setw(14) << random_value << "\n";
    //
    // Each process now carries out TRIAL_NUM trials, and then
    // sends the value back to the master process.
    //
    hit_num = buffon_laplace_simulate(a, b, l, trial_num);

    //
    // initialize sync buffer for reduction
    //
    for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; ++i) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();

    shmem_int_sum_to_all(&hit_total, &hit_num, 1, 0, 0, process_num, pWrk,
                         pSync);

    //
    // The master process can now estimate PI.
    //
    if (process_rank == master) {
        trial_total = trial_num * process_num;

        pdf_estimate = (double) hit_total / (double) trial_total;

        if (hit_total == 0) {
            pi_estimate = r8_huge();
        }
        else {
            pi_estimate = l * (2.0 * (a + b) - l) / (a * b * pdf_estimate);
        }

        pi_error = r8_abs(PI - pi_estimate);

        std::cout << "\n";
        std::cout <<
            "    Trials      Hits    Estimated PDF       Estimated Pi        Error\n";
        std::cout << "\n";
        std::cout << "  " << std::setw(8) << trial_total
         << "  " << std::setw(8) << hit_total
         << "  " << std::setw(16) << pdf_estimate
         << "  " << std::setw(16) << pi_estimate
         << "  " << std::setw(16) << pi_error << "\n";
    }
    //
    // Shut down
    //
    if (process_rank == master) {
        std::cout << "\n";
        std::cout << "BUFFON_LAPLACE - Master process:\n";
        std::cout << "  Normal end of execution.\n";
    }

    shmem_finalize();

    return 0;
}