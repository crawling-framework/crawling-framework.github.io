#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>    // std::swap, std::binary_search

using namespace std;


class IntPair_Set : public set<pair<int, int>>
{
    private:
        map<int, int> node_deg_map;

    public:
        IntPair_Set() {};

        bool inline add(int node, int deg) {
            auto it_res = insert(pair<int, int>(deg, node));
            node_deg_map[node] = deg;
            return it_res.second;
        }

//        /// remove node->deg entry
//        bool remove(int node, int deg){
//            // remove from map if present
//            auto it = node_deg_map.find(node);
//            if (it != node_deg_map.end())
//                node_deg_map.erase(it);
//            // remove from set, return true if success
//            return erase(pair<int, int>(deg, node)) > 0;
//        }

        /// update node entry with a new degree. Returns true if element was replaced.
        bool update(int node, int new_deg) {
            // detect degree in map
            auto it = node_deg_map.find(node);
            if (it != node_deg_map.end()) {
                int old_deg = it->second;
//                node_deg_map.emplace_hint(it, node, new_deg); // TODO may be faster?
                node_deg_map[node] = new_deg;
                erase(pair<int, int>(old_deg, node));
                insert(pair<int, int>(new_deg, node));
                return true;
            }
            node_deg_map[node] = new_deg;
            insert(pair<int, int>(new_deg, node));
            return false;
        }

        pair<int, int> pop() {
            auto r = --end();
            pair<int, int> res = *r;
            erase(r);
            node_deg_map.erase(r->second);
            return res;
        }

        void print_me() {
            printf("set (deg, node): ");
            for (auto it = begin(); it != end(); ++it) {
                std::cout << it->first << ',' << it->second << ' ';
            }
            printf("\nmap (node -> deg): ");
            for(auto n : node_deg_map) {
                std::cout << n.first << "->" << n.second << ' ';
            }
            printf("\n");
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
    l.push_back(pair<int, int>(3, 1));
    l.push_back(pair<int, int>(15, 1));
    l.push_back(pair<int, int>(16, 1));
    l.push_back(pair<int, int>(41, 1));
    l.push_back(pair<int, int>(51, 1));
    l.push_back(pair<int, int>(21, 2));
    l.push_back(pair<int, int>(11, 3));

    IntPair_Set cpq = IntPair_Set();

    for (pair<int, int> e : l) {
        printf("C pushing (%d, %d)\n", e.first, e.second);
        cpq.add(e.first, e.second);
    }

    cpq.print_me();
//    printf("back: %d\n", cpq.back());
//    printf("back: %d\n", cpq.back());
//    printf("back: %d\n", cpq.back());
//

//C updating (11, 1)
//C updating (15, 1)
//C updating (16, 1)
//C updating (41, 1)
//C updating (43, 1)
//C updating (48, 1)
//C updating (11, 2)
//C updating (21, 1)
//C updating (31, 1)
//C updating (43, 2)
//C updating (21, 2)
//C updating (31, 2)
//C updating (3, 1)
//C updating (11, 3)
//C updating (31, 3)
//C updating (51, 1)
//C updating (3, 2)
//C updating (30, 1)
//C updating (41, 2)

    pair<int, int> r;
    r = pair<int, int>(11, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(15, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(16, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(41, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(43, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(48, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(11, 2);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(21, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(31, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(43, 2);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(21, 2);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(31, 2);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(3, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(11, 3);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(31, 3);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(51, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(3, 2);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(30, 1);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    r = pair<int, int>(41, 2);
    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

    printf("updating (%d, %d)\n", r.first, r.second);
    cpq.update(r.first, r.second);
    cpq.print_me();

//    r = pair<int, int>(7, 10);
//    printf("updating (%d, %d)\n", r.first, r.second);
//    cpq.update(r.first, r.second);
//
//    cpq.print_me();

//    while(!cpq.empty()) {
//        int a = cpq.top();
//        cpq.pop();
//        printf("2:%d\n", a);
//    }

//    priority_queue<int, vector<int>, function<bool(int, int)>> cpq = priority_queue<int, vector<int>, function<bool(int, int)>>(cmp);

}
