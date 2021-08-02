#include "bits/stdc++.h"
using namespace std;
#define pii pair<int,int>


inline namespace __dsu {
    vector<int> parent(1e6 + 2047,-1);
    vector<int> cont(1e6 + 2047,0);


    int findp(int v) {
        if (v == parent[v])
            return v;
        return parent[v] = findp(parent[v]);
    }

    void make_set(int v) {
        parent[v] = v;
        cont[v] = 1;
    }

     void make_set_n(int n) {
        for(int i=0; i<=n; i++) {
            make_set(i);
        }
    }

    void unionp(int a, int b) {
        a = findp(a);
        b = findp(b);
        if (a != b) {
            if (cont[a] < cont[b])
                swap(a, b);
            parent[b] = a;
            cont[a] += cont[b];
        }
    }
}

inline namespace __graph {
    struct Graph {
        vector<vector<int>> graph, g_rev;
        vector<int> vis, cost, dp, parent, order, component, deg;
        int n;    
        Graph(int x) {
            n = x - 1;
            graph.resize(x);
            g_rev.resize(x);
            vis.resize(x);
            cost.resize(x);
            parent.resize(x);
            dp.resize(x);
            deg.resize(x);
        }

        void UaddEdge(int u, int v) {
            graph[u].push_back(v);
            graph[v].push_back(u);
            deg[u]++, deg[v]++;
        }

        void DaddEdge(int u, int v) {
            graph[u].push_back(v);
            g_rev[v].push_back(u);
            deg[v]++;
        }

        Graph GenerateReverseGraph() const {
            Graph g_rev(graph.size());
            for (int v = 0; v < graph.size(); v++) {
            for (const auto& e : graph[v]) {
                g_rev.DaddEdge(e, v);
            }
            }
            return g_rev;
        }

        void dfs(int u, int p = 0) {
            vis[u] = 1;
            parent[u] = p;
            for(auto it : graph[u]) {
                if(vis[it] == 0) {
                    dfs(it, u);
                }
            }
            order.push_back(u);
        }

        vector<int> topological_sort() {
            vis.assign(n, false);
            order.clear();
            for (int i = 0; i < n; ++i) {
                if (!vis[i])
                    dfs(i);
            }
            reverse(order.begin(), order.end());
            return order;
        }

        void dfs_scc(int u) {
            vis[u] = 1;
            component.push_back(u);
            for(auto it : g_rev[u]) {
                if(vis[it] == 0) {
                    dfs_scc(it);
                }
            }
        }

        void scc() { 
            for(int i=1; i<=n; i++) {
                if(!vis[i]) {
                    dfs(i);
                }
            }
            reverse(order.begin(), order.end());
            vis.assign((int)vis.size(), 0);
            for(auto v : order) {
                if(!vis[v]) {
                    dfs_scc(v);
                    // component contains scc elemenets
                    component.clear();
                }
            }
        }

        void dfs_cycle(int u, int p) {
            vis[u] = 1;
            for(auto it : graph[u]) {
                if(it != p) { 
                    if(vis[it] == 1) {
                        // cycle;
                        cout << -1 ;
                        exit(0);
                    } else if(vis[it] == 0) {
                        dfs_cycle(it, u);
                    }
                }
            }
            vis[u] = 2;
        }

        void check_cycle() {
            for(int i=1; i<=n; i++) {
                if(vis[i] == 0) {
                    dfs_cycle(i, 0);
                }
            }
            vis.assign(n+1, 0);
        }

    };
}

namespace __DS {
    class Tree {
        public:
            int val;
            Tree *left;
            Tree *right;
    };


    class LLNode {
        public:
            int val;
            LLNode *next;
    };




    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode() : val(0), left(nullptr), right(nullptr) {}
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    };



    struct ListNode {
        int val;
        ListNode *next;
        ListNode() : val(0), next(nullptr) {}
        ListNode(int x) : val(x), next(nullptr) {}
        ListNode(int x, ListNode *next) : val(x), next(next) {}
    };
}
using namespace __DS;


inline namespace __dijkastra {
    const int INF = 1000000000;
    vector<vector<pair<int, int>>> adj;

    void dijkstra(int s, vector<int> & d) {
        int n = adj.size();
        d.assign(n, INF);

        d[s] = 0;
        priority_queue<pii, vector<pii>, greater<pii>> q;
        q.push({0, s});
        while (!q.empty()) {
            int v = q.top().second;
            int d_v = q.top().first;
            q.pop();
            if (d_v != d[v])
                continue;

            for (auto edge : adj[v]) {
                int to = edge.first;
                int len = edge.second;

                if (d[v] + len < d[to]) {
                    d[to] = d[v] + len;
                    q.push({d[to], to});
                }
            }
        }
    }
}