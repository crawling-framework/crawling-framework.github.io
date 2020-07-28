from cpython.mem cimport PyMem_Malloc
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector


cdef extern from "Snap.h":
    # cdef cppclass PUNGraph:
    #     PUNGraph()
    #     PUNGraph New()
    #     int AddEdge(int, int)

    cdef cppclass THash[TKey, TDat]:
        cppclass TIter:
            TDat GetDat()
        THash()
        THashKeyDatI BegI()
        THashKeyDatI EndI()

    cdef cppclass TRnd:
        TRnd()
        TRnd(const int& _Seed, const int& Steps)
        void Randomize()
        void PutSeed(const int& _Seed)
        int GetUniDevInt(const int&)

    cdef cppclass TInt:
        Tint()
        TInt(const int& _Val)
        int operator() () const
        TRnd Rnd

    cdef cppclass TVec[T]:
        TVec()
        void Shuffle(TRnd& Rnd)
        int Add(const T& Val)
        T* BegI() const
        T* EndI() const

    ctypedef TVec[TInt] TIntV

    cdef cppclass TIntPr:
        TIntPr()

    cdef cppclass TFlt:
        TFlt()
        float operator() () const

    cdef cppclass TFltPtr:
        TFltPtr()

    cdef cppclass TStr:
        TStr()
        TStr(const char*)

    # cdef cppclass TCRef:
    #     TCRef()
    #     int GetRefs() const

    cdef cppclass TPt[T]:
        @staticmethod
        TPt New()
        TPt()
        TPt(T*)
        bint Empty() const
        void Clr()
        T operator*()
        bint operator==(const TPt& Pt) const
        T& operator[](const int& RecN) const
        int GetRefs() const

    cdef cppclass THashKeyDatI[TKey, TDat]:
        THashKeyDatI& operator++ ()
        bint operator==(const THashKeyDatI& HashKeyDatI) const
        const TKey& GetKey()
        const TDat& GetDat()

    ctypedef TPt[TUNGraph] PUNGraph
    ctypedef THash[TInt, TFlt] TIntFltH
    ctypedef THash[TIntPr, TFlt] TIntPrFltH

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

        cppclass TEdgeI:
            TEdgeI operator++ ()
            int GetId() const
            int GetSrcNId() const
            int GetDstNId() const
            # int GetDeg()
            # int GetNbrNId(int)

        TUNGraph()
        @staticmethod
        PUNGraph New()
        int AddNode(int)
        int AddEdge(int, int)
        int GetNodes()
        int GetEdges()
        TNodeI GetNI(int)
        TNodeI BegNI() const
        TNodeI EndNI() const
        TEdgeI BegEI() const
        TEdgeI EndEI() const
        bint IsNode(const int)
        bint IsEdge(const int, const int)
        int GetRndNId(TRnd)
        # void GetNIdV(TVec[TInt]& NIdV) const
        void GetNIdV(TIntV& NIdV) const


cdef extern from "Snap.h" namespace "TSnap":
    PGraph LoadEdgeList[PGraph](const TStr&, const int&, const int&)
    double GetClustCf[PGraph](const PGraph& Graph, int SampleNodes)
    double GetMxWccSz[PGraph](const PGraph& Graph)
    PGraph GetMxWcc[PGraph](const PGraph& Graph)
    double GetBfsEffDiam[PGraph](const PGraph& Graph, const int& NTestNodes, const bint& IsDir)
    void GetBetweennessCentr[PGraph](const PGraph& Graph, TIntFltH& NIdBtwH, const double& NodeFrac, const bint& IsDir)
    void GetPageRank[PGraph](const PGraph& Graph, TIntFltH& PRankH, const double& C, const double& Eps, const int& MaxIter)
    double GetClosenessCentr[PGraph](const PGraph& Graph, const int& NId, const bint& Normalized, const bint& IsDir)
    int GetNodeEcc[PGraph](const PGraph& Graph, const int& NId, const bint& IsDir)
    double GetANodeClustCf "TSnap::GetNodeClustCf"[PGraph](const PGraph& Graph, const int& NId)
    void GetNodesClustCf "TSnap::GetNodeClustCf"[PGraph](const PGraph& Graph, TIntFltH& NIdCCfH)
    PGraph GetKCore[PGraph](const PGraph& Graph, const int& K)

    # a
    PUNGraph GenConfModel(const TIntV& DegSeqV, TRnd& Rnd)


# from time import time
cdef TRnd t_random
# t_random.Randomize()
# t_random.PutSeed(2)


cdef class MyGraph:
    # cdef TUNGraph _snap_graph
    cdef PUNGraph _snap_graph_ptr  # TODO extend for directed
    cdef char* _path
    cdef char* _name
    cdef bint _directed
    cdef bint _weighted
    cdef _fingerprint
    cdef _stats_dict

    cdef new_snap(self, PUNGraph snap_graph_ptr, name=?)

    cpdef MyGraph copy(self)

    cpdef giant_component(self)

    cdef PUNGraph snap_graph_ptr(self)

    cpdef bint is_loaded(self)

    cpdef void save(self)

    cpdef void load(self)

    cpdef int nodes(self)

    cpdef int edges(self)

    cpdef bint add_node(self, int node)

    cpdef bint add_edge(self, int i, int j)

    cpdef bint has_node(self, int node)

    cpdef bint has_edge(self, int i, int j)

    cpdef int deg(self, int node)

    cpdef double clustering(self, int node)

    cpdef int max_deg(self)

    cpdef int random_node(self)

    cpdef vector[int] random_nodes(self, int count=?)

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
