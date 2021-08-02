#include "bits/stdc++.h" 
#include <ext/pb_ds/assoc_container.hpp> // Common file 
#include <ext/pb_ds/tree_policy.hpp> 
using namespace __gnu_pbds;
using namespace std;
inline namespace __definitions {
    #define pb push_back
    #define mpr make_pair
    #define pii pair<int, int>
    #define ll long long
    #define ld long double
    #define mint long long
    #define all(arr) arr.begin(), arr.end()
    #define fi first
    #define se second
    #define rep(i, l, r) for(int i=l; i <= r; i++)
    #define FOR(i,n) for(int i=0; i<n; i++)
    #define pie 3.14159265358979323846264338327950L
    #define mid(l, r) l + (r - l) / 2
    #define rev reverse
    #define vi vector<int>
    #define vs vector<string>
    #define vvi vector<vector<int>>
    #define vpii vector<pii>
    #define vvpii vector<vector<pii>>
    #define mii map<int,int>
    #define mci map<char,int>
    #define si set<int>
    #define spii set<pii>
    #define lb lower_bound
    #define ub upper_bound
    #define endl "\n"
    #define sz(x) (int)x.size()
    #define template_array_size (int)1e6 + 6
    #define inf (int)1e9 + 18
    #define sumo(arr) accumulate(all(arr),(ll)0)
    #define mini(arr) *min_element(all(arr))
    #define maxi(arr) *max_element(all(arr))
    #define ref return
    #define trav(x, arr) for(auto &x : arr)
    #define traced(arr)              for(auto it : arr) cout<<it<<" "; cout<<endl;
    #define TRACE
    #ifdef TRACE
    #define trace(...) __f(#__VA_ARGS__, __VA_ARGS__)
        template <typename Arg1>
        void __f(const char* name, Arg1&& arg1){
            cout << name << " : " << arg1 << endl;
            //use cout if u want to display at the bottom
        }
        template <typename Arg1, typename... Args>
        void __f(const char* names, Arg1&& arg1, Args&&... args){
            const char* comma = strchr(names + 1, ','); cout.write(names, comma - names) << " : " << arg1<<" | ";__f(comma+1, args...);
        }
    #else
    #define trace(...)
    #endif

    #define initdp(dp,val,n,m) for(int i=0; i<=n; i++) for(int j=0; j<=m; j++) dp[i][j] = val;
    #define initdp1(dp, val, n) for(int i=0; i<=n; i++)  dp[i]= val;

    int mod = 1e9 + 7;
    int mod2 = 998244353;
    int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};
    int ddx[8]={1,1,0,-1,-1,-1,0,1},ddy[8]={0,1,1,1,0,-1,-1,-1};
    int X[8] = { 2, 1, -1, -2, -2, -1, 1, 2 };  int Y[8] = { 1, 2, 2, 1, -1, -2, -2, -1 };
    template <typename T> void amin(T &a, const T &b) { a = min(a, b); }
    template <typename T> void amax(T &a, const T &b) { a = max(a, b); }
     
    int gcd(int a,int b){ if(!a)return b;return gcd(b%a,a);}
    int lcm(int a, int b) { return (a*b)/ gcd(a,b);}

    typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update>  ord_set;
    typedef tree<pii, null_type, less<pii>, rb_tree_tag,  tree_order_statistics_node_update> ord_pii_set; 

    bool isvalid(int i, int j, int n, int m) {if(i < 0 ||  j < 0 || i  >= n || j >= m) {return false;} return true;}

    ll binpow(ll a, ll b) {a %= mod;ll res = 1;while (b > 0) {if (b & 1)res = res * a % mod;a = a * a % mod;b >>= 1;        }        return res;   }    
    ll mul(ll a, ll b) {    return (( a % mod ) * (b %  mod)% mod + mod) % mod; }  
    ll add(ll a, ll b) {   return (a%mod+ b%mod + 2*mod) % mod; }   
    ll sub(ll a, ll b) {       return (a%mod - b%mod + mod + 2*mod) % mod;    }   
    ll divide(ll a, ll b) {       return  a * (ll)binpow(b,mod-2) % mod;   }

    vector<mint> int_ll(vector<int>& arr) {  vector<mint> nums;  for(auto it : arr) nums.push_back(it);  return nums;   }
    vector<int> ll_int(vector<mint>& arr) { vector<int> nums;   for(auto it : arr) nums.push_back(it);  return nums;   }
}



