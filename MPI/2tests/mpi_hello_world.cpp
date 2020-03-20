#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	constexpr int MAIN_NODE {0};
	constexpr int count {20000};

	// init MPI
	MPI_Init(&argc, &argv);

	// get # prcoesses
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// get ranks
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// print hello world
	printf("hello world from processor %s, rank %d, out of %d processors\n",
		processor_name, world_rank, world_size);


	if (world_rank == MAIN_NODE)
	{
		printf("I AM THE MAIN NODE\n");
	}

	float* local = new float[count];
	float* global = new float[count];

	for(int i = 0; i < count; i++)
	{
		local[i] = world_rank+i;
		global[0] = 0;
	}

	MPI_Allreduce(local, global, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Checking if arrays are the same...\n");
	for(int i = 0; i < count; i++)
	{
		printf("Rank %d data[%d]: %f\n", world_rank, i, global[i]/world_size);
	}


	MPI_Finalize();
	delete[] local;
	delete[] global;

	return 0;
}