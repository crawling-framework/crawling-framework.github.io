from cpython.mem cimport PyMem_Malloc


cdef extern from "Snap.h":
    # cdef cppclass PUNGraph:
    #     PUNGraph()
    #     PUNGraph New()
    #     int AddEdge(int, int)

    cdef cppclass THash[TKey, TDat]:
        cppclass TIter:
            TDat GetDat()

    cdef cppclass TRnd:
        TRnd()
        void Randomize()
        int GetUniDevInt(const int&)

    cdef cppclass TInt:
        Tint()
        TRnd Rnd

    cdef cppclass PUNGraph:
        PUNGraph()
        PUNGraph New()
        int AddNode(int)
        int AddEdge(int, int)
        int GetNodes()
        int GetEdges()

    cdef cppclass TUNGraph:
        cppclass TNode:
            TNode()
            int GetNbrNId(int)

        cppclass TNodeI:
            # ctypedef int THashIter
            # THash[TInt, TNode].TIter NodeHI
            TNodeI operator++ ()
            int GetId()
            int GetDeg()
            int GetNbrNId(int)

        TUNGraph()
        # @staticmethod
        # TUNGraph New()
        int AddNode(int)
        int AddEdge(int, int)
        int GetNodes()
        int GetEdges()
        TNodeI GetNI(int)
        TNodeI BegNI() const
        TNodeI EndNI() const
        bint IsNode(const int)
        int GetRndNId(TRnd)

    cdef cppclass TStr:
        TStr()
        TStr(const char*)

    cdef cppclass TPt[T]:
        TPt()
        T operator*()

cdef extern from "Snap.h" namespace "TSnap":
    PGraph LoadEdgeList[PGraph](const TStr&, const int&, const int&)


cdef class CGraph:
    cdef TUNGraph _snap_graph  # TODO extend for directed
    cdef char* _path
    cdef char* _name
    cdef bint _directed
    cdef bint _weighted
    cdef _fingerprint
    cdef _stats_dict

    cdef CGraph load(self)

    cpdef int nodes(self)

    cpdef int edges(self)

    cpdef bint add_node(self, int node)

    cpdef bint add_edge(self, int i, int j)

    cpdef bint has_node(self, int node)

    cpdef int deg(self, int node)

    cpdef int random_node(self)

    cpdef int random_neighbor(self, int node)


cdef inline char* str_to_chars(str string):
    cdef int length = len(string)
    cdef char* res = <char *>PyMem_Malloc((length+1) * sizeof(char))
    if not res:  # as in docs, a good practice
        raise MemoryError()

    b_string = string.encode()
    for i in range(length):
        res[i] = <char>(b_string[i])
    res[length] = '\0'
    return res
