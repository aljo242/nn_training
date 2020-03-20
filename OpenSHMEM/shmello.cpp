#include <stdio.h>
#include <shmem.h>

int main(void)
{
	shmem_init();
	int npes = shmem_n_pes();
	int me = shmem_my_pe();

	printf("Hello from %d of %d\n", me, npes);	

	return 0;
}
