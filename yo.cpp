/* SarcasticMonk - Vipul Bansal */

#pragma GCC optimize("Ofast")
#pragma GCC target ("avx2")
#pragma GCC optimize("unroll-loops")
#include "bits/stdc++.h"
#include <ext/pb_ds/assoc_container.hpp> 
#include <ext/pb_ds/tree_policy.hpp> 
using namespace __gnu_pbds;
using namespace std;

inline namespace FileIO {
    void setIO() {
        ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
    cout << fixed << setprecision(2);
        #ifndef ONLINE_JUDGE    
            // freopen("input.txt","r",stdin);
            // freopen("output.txt","w",stdout); 
        #endif
    }
}

inline namespace __SarcasticMonk {

    // definitions
    #define pb push_back
    #define mpr make_pair
    #define pii pair<int, int>
    #define ll long long
    #define ld long double
    #define all(arr) arr.begin(), arr.end()
    #define fi first
    #define se second
    #define rep(i, l, r) for(int i=l; i <= r; i++)
    #define FOR(i,n) for(int i=0; i<n; i++)
    #define pie 3.14159265358979323846264338327950L
    #define mid(l, r) l + (r - l) / 2
    #define ret return
    #define rev reverse
    #define nl "\n"
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
    #define YES cout << "YES" << "\n";
    #define NO cout << "NO" << "\n";
    #define Yes cout << "Yes" << "\n";
    #define No cout << "No" << "\n";
    #define yes cout << "yes" << "\n";
    #define no cout << "no" << "\n";
    #define endl "\n"
    #define int long long
    #define sz(x) (int)x.size()
    #define INF (int)1e18 + 9
    #define inf (int)1e9 + 18
    #define ret return
    #define trav(x, arr) for(auto &x : arr)
    #define traced(arr)         for(auto it : arr) cerr<<it<<" "; cerr<<endl;
    #define TRACE
    #ifdef TRACE
    #define trace(...) __f(#__VA_ARGS__, __VA_ARGS__)
        template <typename Arg1>
        void __f(const char* name, Arg1&& arg1){
            cerr << name << " : " << arg1 << endl;
            //use cerr if u want to display at the bottom
        }
        template <typename Arg1, typename... Args>
        void __f(const char* names, Arg1&& arg1, Args&&... args){
            const char* comma = strchr(names + 1, ','); cerr.write(names, comma - names) << " : " << arg1<<" | ";__f(comma+1, args...);
        }
    #else
    #define trace(...)
    #endif


    int gcd(int a,int b){ if(!a)return b;return gcd(b%a,a);}
    int lcm(int a, int b) { return (a*b)/ gcd(a,b);}


    #define initdp(dp,val,n,m) for(int i=0; i<=n; i++) for(int j=0; j<=m; j++) dp[i][j] = val;
    #define initdp1(dp, val, n) for(int i=0; i<=n; i++)  dp[i]= val;

    typedef tree<int, null_type, less<int>, rb_tree_tag,tree_order_statistics_node_update> ord_set;
    typedef tree<pii, null_type, less<pii>, rb_tree_tag,tree_order_statistics_node_update> ord_pii_set;

    // Constants
    int mod = 1e9 + 7;
    int mod2 = 998244353;
    int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};
    int ddx[8]={1,1,0,-1,-1,-1,0,1},ddy[8]={0,1,1,1,0,-1,-1,-1};
    template <typename T> void amin(T &a, const T &b) { a = min(a, b); }
    template <typename T> void amax(T &a, const T &b) { a = max(a, b); }
    template<typename T1,typename T2>istream& operator>>(istream& in,pair<T1,T2> &a){in>>a.fr>>a.sc;return in;}
    template<typename T1,typename T2>ostream& operator<<(ostream& out,pair<T1,T2> a){out<<a.fr<<" "<<a.sc;return out;}

    // Output


    template <typename T> struct is_outputtable { template <typename C> static constexpr decltype(declval<ostream &>() << declval<const C &>(), bool()) test(int32_t) { return true; } template <typename C> static constexpr bool test(...) { return false; } static constexpr bool value = test<T>(int32_t()); };
    template <class T, typename V = decltype(declval<const T &>().begin()), typename S = typename enable_if<!is_outputtable<T>::value, bool>::type> void pr(const T &x);
    template <class T, typename V = decltype(declval<ostream &>() << declval<const T &>())> void pr(const T &x) { cout << x; }
    template <class T1, class T2> void pr(const pair<T1, T2> &x);
    template <class Arg, class... Args> void pr(const Arg &first, const Args &...rest) { pr(first); pr(rest...); }
    template <class T, bool pretty = false> void prContain(const T &x) { if (pretty) pr("{"); bool fst = 1; for (const auto &a : x) pr(!fst ? pretty ? " " : " " : "", a), fst = 0; if (pretty) pr("}"); }
    template <class T> void pc(const T &x) { prContain<T, false>(x); pr("\n"); }
    template <class T1, class T2> void pr(const pair<T1, T2> &x) { pr( x.first, " ", x.second, "\n" );  }
    template <class T, typename V, typename S> void pr(const T &x) { prContain(x); }
    void out() { pr("\n"); }
    template <class Arg> void out(const Arg &first) { pr(first); out(); }
    template <class Arg, class... Args> void out(const Arg &first, const Args &...rest) { pr(first, " "); out(rest...); }

    // Input

    template <class T1, class T2> void re(pair<T1, T2> &p);
    template <class T> void re(vector<T> &a);
    template <class T, size_t SZ> void re(array<T, SZ> &a);
    template <class T> void re(T &x) { cin >> x; }
    void re(double &x) { string t; re(t); x = stod(t); }
    template <class Arg, class... Args> void re(Arg &first, Args &...rest) { re(first); re(rest...); }
    template <class T1, class T2> void re(pair<T1, T2> &p) { re(p.first, p.second); }
    template <class T> void re(vector<T> &a) { for (int32_t i = 0; i < sz(a); i++) re(a[i]); }
    template <class T, size_t SZ> void re(array<T, SZ> &a) { for (int32_t i = 0; i < SZ; i++) re(a[i]); }

    // MODULAR OPERATIONS

    int binpow(int a, int b) {a %= mod;int res = 1;while (b > 0) {if (b & 1) res = res * a % mod; a = a * a % mod; b >>= 1; }return res;}
    int mul(int a, int b) {return (( (a + mod)% mod ) * ((b+mod) %  mod)) % mod;}   
    int add(int a, int b) { return (a%mod+ b%mod + 2*mod) % mod;}   
    int sub(int a, int b) { return (a%mod - b%mod + 2*mod) % mod;}
    int divide(int a, int b) {return  (a+mod)%mod * (int)binpow((b+mod)%mod,mod-2) % mod; }
};

// start code from here on .......







void test_case(int tc) {  



}
 



int32_t main ()
{
    setIO();

    int t = 1;
    cin >> t;
    for(int i=1; i<=t; i++) {
        // cout << "Case #"<< i << ": " ;
        test_case(i);
    }
    // cerr << "\nTime elapsed: " << 1000 * clock() / CLOCKS_PER_SEC << "ms\n";

}
