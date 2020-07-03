#include <iostream>
#include "mpi.h"

using namespace std;

int main(int argc, char** argv){
	int rank, size;
    MPI_Status Status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int msg = 1;
	int src = (rank + 1) % size;
	if (!rank){
		MPI_Send(&msg, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
		cout << rank << " rank send to " << src << endl;
		MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
		cout << rank << " rank recv from " << size - 1 << endl;
	}
	else{
		MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
		cout << rank << " rank recv from " << rank - 1 << endl;
		MPI_Send(&msg, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
		cout << rank << " rank send to " << src << endl;
	}
	MPI_Finalize();

	return 0;
}