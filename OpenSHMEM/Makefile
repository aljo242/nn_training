SHMEM_CXX      = oshcxx
SHMEM_CXXFLAGS = -g -O2

CXX_TESTS      = $(wildcard *.cxx)

CXX_EXES       = $(CXX_TESTS:.cxx=.x)

EXES           = $(CXX_EXES)

.PHONY: clean all default

all default:	$(EXES)

.SUFFIXES: .x

%.x:	%.cxx
	$(SHMEM_CXX) $(SHMEM_CXXFLAGS) -o $@ $^

clean:
	rm -f $(EXES)