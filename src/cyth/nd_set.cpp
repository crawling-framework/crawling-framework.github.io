#include <iostream>
#include <vector>
#include <set>
#include <algorithm>    // std::swap, std::binary_search

using namespace std;


class IntPair_Set : public set<pair<int, int>>
{
    public:
        IntPair_Set() {};

        bool inline add(int node, int deg) {
            auto it_res = insert(pair<int, int>(deg, node));
            return it_res.second;
        }

        bool remove(int node, int deg){
            return erase(pair<int, int>(deg, node));
        }

        pair<int, int> pop() {
            auto r = --end();
            pair<int, int> res = *r;
            erase(r);
            return res;
        }
};

//bool cmp(int a, int b)
//{
//    printf("C cmp %d and %d\n", a, b);
//    return a < b;
//}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("123\n");
    }
    vector<pair<int, int>> l;
//    for(int i=1; i<10; ++i) {
//        l.push_back(pair<int, int>(i, 10*i));
//    }
    l.push_back(pair<int, int>(3, 30));
    l.push_back(pair<int, int>(5, 10));
    l.push_back(pair<int, int>(4, 10));
    l.push_back(pair<int, int>(1, 10));
    l.push_back(pair<int, int>(2, 20));

//    priority_queue<int, vector<int>, function<bool(int, int)>> cpq = priority_queue<int, vector<int>, function<bool(int, int)>>(cmp);
    IntPair_Set cpq = IntPair_Set();

    for (pair<int, int> e : l) {
        printf("C pushing (%d, %d)\n", e.first, e.second);
        cpq.add(e.first, e.second);
    }

//    for (int i : l)
//        printf("%d, ", i);
//
//    int e = 10;
//    printf("pos of %d is %d\n", e, find<int>(l, e, cmp));


//    while(!cpq.empty()) {
//        int a = cpq.top();
//        cpq.pop();
//        printf("1:%d\n", a);
//    }

    for(pair<int, int> n : cpq) {
        std::cout << n.first << ',' << n.second << ' ';
    }
//    printf("back: %d\n", cpq.back());
//    printf("back: %d\n", cpq.back());
//    printf("back: %d\n", cpq.back());
//
    pair<int, int> r = pair<int, int>(3, 10);
    printf("removing (%d, %d)\n", r.first, r.second);
    cpq.remove(r.first, r.second);
//    pair<int, int> a = cpq.pop();

    for(pair<int, int> n : cpq) {
        std::cout << n.first << ',' << n.second << ' ';
    }

//    while(!cpq.empty()) {
//        int a = cpq.top();
//        cpq.pop();
//        printf("2:%d\n", a);
//    }

//    priority_queue<int, vector<int>, function<bool(int, int)>> cpq = priority_queue<int, vector<int>, function<bool(int, int)>>(cmp);

}
