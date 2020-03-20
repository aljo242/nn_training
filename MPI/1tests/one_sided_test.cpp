#include <stdio.h>
#include <mpi.h>

constexpr int NUM_ELEMENT {1000};

void AvgReduce_float(float* sendData, const int size, const int num_procs, MPI_Win win);

int main(int argc, char** argv)
{
	int i, id, num_procs, len;
	float* localbuffer = new float[NUM_ELEMENT];
	float* sharedbuffer = new float[NUM_ELEMENT];
	char name[MPI_MAX_PROCESSOR_NAME];

	// MPI initializations
	MPI_Win win;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Get_processor_name(name, &len);

	// rank print
	printf("Rank %d running on %s\n", id, name);

	MPI_Win_create(sharedbuffer, NUM_ELEMENT*sizeof(float), sizeof(float), MPI_INFO_NULL,
		MPI_COMM_WORLD, &win);

	// init these vals before communication
	for (i = 0; i < NUM_ELEMENT; i++)
	{
		sharedbuffer[i] = 10.0f*id + i;
		localbuffer[i]  = 0.0f; 
	}

	printf("Rank %d sets data in the shared memory: ", id);

	for (i = 0; i < NUM_ELEMENT; i++)
	{
		printf(" %02f", sharedbuffer[i]);
	}
	printf("\n");


	MPI_Win_fence(0, win);
	if (id != 0)
	{
		MPI_Get(&localbuffer[0], NUM_ELEMENT, MPI_FLOAT, id-1, 0, NUM_ELEMENT, MPI_FLOAT, win);
	}
	else
	{
		MPI_Get(&localbuffer[0], NUM_ELEMENT, MPI_FLOAT, num_procs - 1, 0, NUM_ELEMENT, MPI_FLOAT, win);
	}
	MPI_Win_fence(0, win);


	printf("Rank %d gets data from the shared memory: ", id);
	for (i = 0; i < NUM_ELEMENT; i++)
	{
		printf(" %02f", localbuffer[i]);
	}
	printf("\n");


	MPI_Win_fence(0, win);
	if (id != num_procs -1)
	{
		MPI_Put(&localbuffer[0], NUM_ELEMENT, MPI_FLOAT, id+1, 0, NUM_ELEMENT, MPI_FLOAT, win);
	}
	else
	{
		MPI_Put(&localbuffer[0], NUM_ELEMENT, MPI_FLOAT, 0, 0, NUM_ELEMENT, MPI_FLOAT, win);
	}
	MPI_Win_fence(0, win);


	printf("Rank %d has new data from the shared memory: ", id);
	for (i = 0; i < NUM_ELEMENT; i++)
	{
		printf(" %02f", sharedbuffer[i]);
	}

	MPI_Barrier(MPI_COMM_WORLD);


	printf("\n");
	for (i = 0; i < NUM_ELEMENT; i++)
	{
		sharedbuffer[i] = 0.0f;
		localbuffer[i]  = 10.0f*id + i; 
	}


	AvgReduce_float(localbuffer, NUM_ELEMENT, num_procs, win);

/*
	for(int i = 0; i < num_procs; i++)
	{
		MPI_Win_fence(0, win);
		{
			MPI_Accumulate(&localbuffer[0], NUM_ELEMENT, MPI_FLOAT, i, 0, NUM_ELEMENT, MPI_FLOAT, MPI_SUM, win);
		}
		MPI_Win_fence(0, win);
	}
*/
	printf("Rank shared memory %d after all sum: ", id);
	for (i = 0; i < NUM_ELEMENT; i++)
	{
		//sharedbuffer[i] = sharedbuffer[i] / num_procs;
		printf(" %02f", sharedbuffer[i]);
	}
	printf("\n");


	MPI_Win_free(&win);
	MPI_Finalize();

	delete[] localbuffer;
	delete[] sharedbuffer;

	return 0;
}


void AvgReduce_float(float* sendData, const int size, const int num_procs, MPI_Win win)
{
	for(int i = 0; i < num_procs; i++)
	{
		MPI_Win_fence(0, win);
		{
			MPI_Accumulate(&sendData[0], size, MPI_FLOAT, i, 0, size, MPI_FLOAT, MPI_SUM, win);
		}
		MPI_Win_fence(0, win);
	}
	for (int i = 0; i < size; i++)
	{
		sendData[i] = sendData[i] / num_procs;
	}
}