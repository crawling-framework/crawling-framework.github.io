#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>    // std::swap, std::binary_search
//#include <stdlib>       // rand, RAND_MAX

using namespace std;

unsigned long long seed = (unsigned long long) time(0);

float inline rnd_01() {
    return float(rand()) / RAND_MAX;
}

class IntPair_Set : public set<pair<int, int>>
{
    public:
        IntPair_Set() {
            srand(time(NULL));
        };

        bool inline add(int node, int deg) {
            auto it_res = insert(pair<int, int>(deg, node));
            return it_res.second;
        }

        /// update degree of node entry by 1. Returns true if element was replaced.
        bool update_1(int node, int deg) {
            int r = erase(pair<int, int>(deg, node));
            insert(pair<int, int>(deg+1, node));
            return r > 0;
        }

        void remove(int node, int deg) {
            erase(pair<int, int>(deg, node));
        }

        pair<int, int> pop() {
            auto r = --end();
            pair<int, int> res = *r;
            erase(r);
            return res;
        }

        void print_me() {
            printf("set (deg, node): ");
            for (auto it = begin(); it != end(); ++it) {
                std::cout << it->first << ',' << it->second << ' ';
            }
            printf("\n");
        }

        /// Sample a random node with probability proportional to node degree, and remove it.
        /// Complexity - O(n)
        pair<int, int> pop_proportional_degree() {
            int len = size();
            if (len == 0) {printf("ERROR: size = 0.\n");}
            int nodes[len];
            int degs[len];
            float cums[len];
            int i = 0; // index
            float cum = 0; // cumulative degree
            for (auto it = begin(); it != end(); ++it) {
                degs[i] = it->first;
                nodes[i] = it->second;
                cum = cum + it->first;
                cums[i] = cum;
                ++i;
            }

            // random * sum of degrees to normalize
            float r = cums[len-1] * rnd_01();
            // bin search
            int ix = std::lower_bound(&(cums[0]), cums+len, r) - cums;
            pair<int, int> res = pair<int, int>(degs[ix], nodes[ix]);
            erase(res);
            return res;
        }
};


class IntPair_Set_With_Map : public set<pair<int, int>>
{
    private:
        map<int, int> node_deg_map;

    public:
        IntPair_Set_With_Map() {};

        bool inline add(int node, int deg) {
            auto it_res = insert(pair<int, int>(deg, node));
            node_deg_map[node] = deg;
            return it_res.second;
        }

        /// update node entry with a new degree. Returns true if element was replaced.
        bool update(int node, int new_deg) {
            // detect degree in map
            auto it = node_deg_map.find(node);
            if (it != node_deg_map.end()) {
                int old_deg = it->second;
//                node_deg_map.emplace_hint(it, node, new_deg); // may be faster?
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

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("123\n");
    }
    vector<pair<int, int>> l;
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

    for (int i=0; i<10; ++i) {
        auto e = cpq.pop_proportional_degree();
        cout << e.first << ',' << e.second << '\n';
    }

}
