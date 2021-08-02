#include "bits/stdc++.h"
#define ll long long
// #define mod 1000000007
using namespace std;
const int N = 3e5;
int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};  
int ddx[8]={1,1,0,-1,-1,-1,0,1},ddy[8]={0,1,1,1,0,-1,-1,-1};
int gcd(int a,int b){ if(!a)return b;return gcd(b%a,a);}
int lcm(int a, int b) { return (a*b)/ gcd(a,b);}

// All possible moves of a knight 
int X[8] = { 2, 1, -1, -2, -2, -1, 1, 2 }; 
int Y[8] = { 1, 2, 2, 1, -1, -2, -2, -1 }; 


// allow recursion for lamba functions in c++
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0200r0.html
template<class Fun> class y_combinator_result {
    Fun fun_;
public:
    template<class T> explicit y_combinator_result(T &&fun): fun_(std::forward<T>(fun)) {}
    template<class ...Args> decltype(auto) operator()(Args &&...args) { return fun_(std::ref(*this), std::forward<Args>(args)...); }
};
template<class Fun> decltype(auto) y_combinator(Fun &&fun) { return y_combinator_result<std::decay_t<Fun>>(std::forward<Fun>(fun)); }
 



// #include <ext/pb_ds/assoc_container.hpp> 
// #include <ext/pb_ds/tree_policy.hpp> 
// using namespace __gnu_pbds;

// if(i < 0 | j < 0 || i >= n || j >= m || matrix[i][j] == 0 || vis[i][j] == 1) {
//         return;
// }

namespace __primecheck {
    unsigned mod_pow(unsigned a, unsigned b, unsigned mod) {
        unsigned result = 1;

        while (b > 0) {
            if (b & 1)
                result = unsigned(uint64_t(result) * a % mod);

            a = unsigned(uint64_t(a) * a % mod);
            b >>= 1;
        }

        return result;
    }

    bool miller_rabin(unsigned n) {
        if (n < 2)
            return false;

        // Check small primes.
        for (unsigned p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29})
            if (n % p == 0)
                return n == p;

        int r = __builtin_ctz(n - 1);
        unsigned d = (n - 1) >> r;

        // https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Testing_against_small_sets_of_bases
        for (unsigned a : {2, 7, 61}) {
            unsigned x = mod_pow(a % n, d, n);

            if (x <= 1 || x == n - 1)
                continue;

            for (int i = 0; i < r - 1 && x != n - 1; i++)
                x = unsigned(uint64_t(x) * x % n);

            if (x != n - 1)
                return false;
        }

        return true;
    }

    int find_prime(int n) {
        while (!miller_rabin(n))
            n++;

        return n;
    }
};
using namespace __primecheck;

// Galen Collins segment tree ( easy to understand but fix merge operation)




inline namespace __segTree {
    template <typename num_t> 
    struct segtree {
      int n, depth;
      vector<num_t> tree, lazy;

      void init(int s, vector<int>& arr) {
        n = s;
        tree = vector<num_t>(4 * s, 0);
        lazy = vector<num_t>(4 * s, 0);
        init(0, 0, n - 1, arr);
      }

      num_t init(int i, int l, int r, vector<int>& arr) {
        if (l == r) return tree[i] = arr[l];

        int mid = (l + r) / 2;
        num_t a = init(2 * i + 1, l, mid, arr),
              b = init(2 * i + 2, mid + 1, r, arr);
        return tree[i] = a.op(b);
      }

      void update(int l, int r, num_t v) {
        if (l > r) return;
        update(0, 0, n - 1, l, r, v);
      }

      num_t update(int i, int tl, int tr, int ql, int qr, num_t v) {
        eval_lazy(i, tl, tr);
        
        if (tl > tr || tr < ql || qr < tl) return tree[i];
        if (ql <= tl && tr <= qr) {
          lazy[i] = lazy[i].val + v.val;
          eval_lazy(i, tl, tr);
          return tree[i];
        }
        
        if (tl == tr) return tree[i];

        int mid = (tl + tr) / 2;
        num_t a = update(2 * i + 1, tl, mid, ql, qr, v),
              b = update(2 * i + 2, mid + 1, tr, ql, qr, v);
        return tree[i] = a.op(b);
      }

      num_t query(int l, int r) {
        if (l > r) return num_t::null_v;
        return query(0, 0, n-1, l, r);
      }

      num_t query(int i, int tl, int tr, int ql, int qr) {
        eval_lazy(i, tl, tr);
        
        if (ql <= tl && tr <= qr) return tree[i];
        if (tl > tr || tr < ql || qr < tl) return num_t::null_v;

        int mid = (tl + tr) / 2;
        num_t a = query(2 * i + 1, tl, mid, ql, qr),
              b = query(2 * i + 2, mid + 1, tr, ql, qr);
        return a.op(b);
      }

      void eval_lazy(int i, int l, int r) {
        tree[i] = tree[i].lazy_op(lazy[i], (r - l + 1));
        if (l != r) {
          lazy[i * 2 + 1] = lazy[i].val + lazy[i * 2 + 1].val;
          lazy[i * 2 + 2] = lazy[i].val + lazy[i * 2 + 2].val;
        }

        lazy[i] = num_t();
      }
    };




    struct nums_t {
      long long val;
      static const long long null_v = 0;

      nums_t(): val(0) {}
      nums_t(long long v): val(v) {}

      nums_t op(nums_t& other) {
        return nums_t(val + other.val);
      }
      
      nums_t lazy_op(nums_t& v, int size) {
        return nums_t(val + v.val);
      }
    };


    // segtree<nums_t> st;
}




string decimalToBinary(int n) 
{   
    string s = bitset<64> (n).to_string();
    return s;
}


bool isVowel(char x) {
    return x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u';
}


const int mod = 1e9 + 7; 

// a power to b mod m
// (a^b) % m

int binpow(int a, int b) {
    a %= mod;
    int res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}
 
int mul(int a, int b) {
    return ( a % mod ) * (b %  mod) % mod;
}
 
int add(int a, int b) {
    return (a%mod+ b%mod) % mod;
}
 
int sub(int a, int b) {
    return (a%mod - b%mod + mod) % mod;
}
 
int divide(int a, int b) {
    return  a * (int)binpow(b,mod-2) % mod;
}



struct Sieve {
  int n;
  vector<int> f, primes;
  Sieve(int n=2e6 + 10):n(n), f(n+1) {
    f[0] = f[1] = -1;
    for (ll i = 2; i <= n; ++i) {
      if (f[i]) continue;
      primes.push_back(i);
      f[i] = i;
      for (ll j = i*i; j <= n; j += i) {
        if (!f[j]) f[j] = i;
      }
    }
  }
  bool isPrime(int x) { return f[x] == x;}
  vector<int> factorList(int x) {
    vector<int> res;
    while (x != 1) {
      res.push_back(f[x]);
      x /= f[x];
    }
    return res;
  }

  vector<pair<int,int> > factor(int x) {
    vector<int> fl = factorList(x);
    if (fl.size() == 0) return vector<pair<int,int> >();
    vector<pair<int,int> > res(1, make_pair(fl[0], 0));
    for (int p : fl) {
      if (res.back().first == p) {
        res.back().second++;
      } else {  
        res.push_back(make_pair(p, 1));
      }
    }
    return res;
  }
};




bool isprime(ll n) //Time Complexity--->sqrt(n)
{   if (n <= 1)  return false; if (n <= 3)  return true; 
  	if (n%2 == 0 || n%3 == 0) return false; 
    for (int i=5; i*i<=n; i=i+6) if (n%i == 0 || n%(i+2) == 0) return false; 
    return true; 
} 


bool ispower2(int x) {
    return x && (!(x&(x-1)));
}




// ncr of large numbers


const int MAX = 2e6 + 10;
long long fac[MAX], finv[MAX], inv[MAX];
 
void COMinit() {
  fac[0] = fac[1] = 1;
  finv[0] = finv[1] = 1;
  inv[1] = 1;
  for (int i = 2; i < MAX; i++){
    fac[i] = fac[i - 1] * i % mod;
    inv[i] = mod - inv[mod%i] * (mod / i) % mod;
    finv[i] = finv[i - 1] * inv[i] % mod;
  }
}

long long COM(int n, int k){
  if (n < k) return 0;
  if (n < 0 || k < 0) return 0;
  return fac[n] * (finv[k] * finv[n - k] % mod) % mod;
}





// Number of factors


const int MAX = 1000001; 
int factor[MAX] = { 0 }; 
  

void generatePrimeFactors() 
{ 
    factor[1] = 1; 
  
    // Initializes all the positions with their value. 
    for (int i = 2; i < MAX; i++) 
        factor[i] = i; 
  
    // Initializes all multiples of 2 with 2 
    for (int i = 4; i < MAX; i += 2) 
        factor[i] = 2; 
  
    // A modified version of Sieve of Eratosthenes to 
    // store the smallest prime factor that divides 
    // every number. 
    for (int i = 3; i * i < MAX; i++) { 
        // check if it has no prime factor. 
        if (factor[i] == i) { 
            // Initializes of j starting from i*i 
            for (int j = i * i; j < MAX; j += i) { 
                // if it has no prime factor before, then 
                // stores the smallest prime divisor 
                if (factor[j] == j) 
                    factor[j] = i; 
            } 
        } 
    } 
} 
 
// function to calculate number of factors 
int calculateNoOFactors(int n) 
{ 
    if (n == 1) 
        return 1; 
  
    int ans = 1; 
  

    int dup = factor[n]; 

    int c = 1; 
  

    int j = n / factor[n]; 

    while (j != 1) { 

        if (factor[j] == dup) 
            c += 1; 

        else { 
            dup = factor[j]; 
            ans = ans * (c + 1); 
            c = 1; 
        } 
  
        j = j / factor[j]; 
    } 
   
    ans = ans * (c + 1); 
  
    return ans; 
} 
 


// LIS in Nlogn

int LIS(vector<int> array, int n)  {
    vector<int > ans;
    for (int i = 0; i < n; i++) {
        int x = array[i];
        //change lower_bound to upper_bound if strictly increasing is not important
        vector<int>::iterator it = lower_bound(ans.begin(), ans.end(), x);
        if (it == ans.end())    {
            ans.push_back(x);
        } 
        else {
            *it = x;
        }
    }
    return ans.size();
} 

int LCS(string X, string Y, int m, int n){
    int L[m + 1][n + 1];
    int i, j;
    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[i][j] = 0;
  
            else if (X[i - 1] == Y[j - 1])
                L[i][j] = L[i - 1][j - 1] + 1;
  
            else
                L[i][j] = max(L[i - 1][j], L[i][j - 1]);
        }
    }
    return L[m][n];
}


vector<int> compute_lps(string s) 
{ 
    int n = s.size(); 
  
    // To store longest prefix suffix 
    vector<int> lps(n); 
  
    // Length of the previous 
    // longest prefix suffix 
    int len = 0; 
  
    // lps[0] is always 0 
    lps[0] = 0; 
    int i = 1; 
  
    // Loop calculates lps[i] for i = 1 to n - 1 
    while (i < n) { 
        if (s[i] == s[len]) { 
            len++; 
            lps[i] = len; 
            i++; 
        } 
        // (pat[i] != pat[len]) 
        else { 
            if (len != 0) 
                len = lps[len - 1]; 
            // Also, note that we do not increment 
            // i here 
  
            // If len = 0 
            else { 
                lps[i] = 0; 
                i++; 
            } 
        } 
    } 
    // ans is the back of the array
    return lps; 
} 


// new lps
vector<int> lps(string S) {
    vector<int> z(S.size());
    int l = -1, r = -1;
    for(int i=1; i < S.size(); ++i){
        z[i] = i >= r ? 0 : min(r - i, z[i - l]);
        while (i + z[i] < S.size() && S[i + z[i]] == S[z[i]])
            z[i]++;
        if (i + z[i] > r)
            l = i, r = i + z[i];
    }
    return z;
}


// Sum queries in Segment Tree


int n, t[4*N];

void build(int a[], int v, int tl, int tr) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = t[v*2] + t[v*2+1];
    }
}

int sum(int v, int tl, int tr, int l, int r) {
    if (l > r) 
        return 0;
    if (l == tl && r == tr) {
        return t[v];
    }
    int tm = (tl + tr) / 2;
    return sum(v*2, tl, tm, l, min(r, tm))
           + sum(v*2+1, tm+1, tr, max(l, tm+1), r);
}

void update(int v, int tl, int tr, int pos, int new_val) {
    if (tl == tr) {
        t[v] = new_val;
    } else {
        int tm = (tl + tr) / 2;
        if (pos <= tm)
            update(v*2, tl, tm, pos, new_val);
        else
            update(v*2+1, tm+1, tr, pos, new_val);
        t[v] = t[v*2] + t[v*2+1];
    }
}



// Lazy 


void build(int a[], int v, int tl, int tr) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = 0;
    }
}

void update(int v, int tl, int tr, int l, int r, int add) {
    if (l > r)
        return;
    if (l == tl && r == tr) {
        t[v] += add;
    } else {
        int tm = (tl + tr) / 2;
        update(v*2, tl, tm, l, min(r, tm), add);
        update(v*2+1, tm+1, tr, max(l, tm+1), r, add);
    }
}

int get(int v, int tl, int tr, int pos) {
    if (tl == tr)
        return t[v];
    int tm = (tl + tr) / 2;
    if (pos <= tm)
        return t[v] + get(v*2, tl, tm, pos);
    else
        return t[v] + get(v*2+1, tm+1, tr, pos);
}


// RMQ

int minVal(int x, int y) { return (x < y)? x: y; }   
int getMid(int s, int e) { return s + (e -s)/2; }  
int RMQUtil(int *st, int ss, int se, int qs, int qe, int index)  
{  
    // If segment of this node is a part  
    // of given range, then return  
    // the min of the segment  
    if (qs <= ss && qe >= se)  
        return st[index];  
  
    // If segment of this node 
    // is outside the given range  
    if (se < qs || ss > qe)  
        return INT_MAX;  
  
    // If a part of this segment 
    // overlaps with the given range  
    int mid = getMid(ss, se);  
    return minVal(RMQUtil(st, ss, mid, qs, qe, 2*index+1),  
                RMQUtil(st, mid+1, se, qs, qe, 2*index+2));  
}   
int RMQ(int *st, int n, int qs, int qe)  
{  
    // Check for erroneous input values  
    if (qs < 0 || qe > n-1 || qs > qe)  
    {  
        cout<<"Invalid Input";  
        return -1;  
    }  
  
    return RMQUtil(st, 0, n-1, qs, qe, 0);  
}  
int constructSTUtil(int arr[], int ss, int se, int *st, int si)  
{  
    // If there is one element in array, 
    // store it in current node of  
    // segment tree and return  
    if (ss == se)  
    {  
        st[si] = arr[ss];  
        return arr[ss];  
    }  
  
    // If there are more than one elements,  
    // then recur for left and right subtrees  
    // and store the minimum of two values in this node  
    int mid = getMid(ss, se);  
    st[si] = minVal(constructSTUtil(arr, ss, mid, st, si*2+1),  
                    constructSTUtil(arr, mid+1, se, st, si*2+2));  
    return st[si];  
}  
int *constructST(int arr[], int n)  
{  
    // Allocate memory for segment tree  
  
    //Height of segment tree  
    int x = (int)(ceil(log2(n)));  
  
    // Maximum size of segment tree  
    int max_size = 2*(int)pow(2, x) - 1;  
  
    int *st = new int[max_size];  
  
    // Fill the allocated memory st  
    constructSTUtil(arr, 0, n-1, st, 0);  
  
    // Return the constructed segment tree  
    return st;  
}  




// query l to r range for the no of integers between x and y


// #include <iostream>
// using namespace std;
// int T = 1;
// const int N = 1e6;
// const int MX = N;
// struct node{
// 	int l, r, cnt;	
// }t[100*MX];
// int root[N], a[N];
// int build(int lo, int hi){
// 	int id = T++;
// 	if(lo == hi) return id;
// 	int  mid = (lo+hi)/2;
// 	t[id].l = build(lo, mid);
// 	t[id].r = build(mid+1, hi);
// 	return id;
// }
// int update(int rt, int lo, int hi, int val){
// 	int id = T++;
// 	t[id] = t[rt]; t[id].cnt++;
// 	if(lo == hi) return id;
// 	int mid = (lo+hi)/2;
// 	if(val <= mid) t[id].l = update(t[rt].l, lo, mid, val);
// 	else t[id].r = update(t[rt].r, mid+1, hi, val);
// 	return id;
// }
// int query(int rt, int lo, int hi, int x, int y){
// 	if(x==lo and y==hi) return t[rt].cnt;
// 	int mid = (lo+hi)/2;
// 	if(y <= mid) return query(t[rt].l, lo, mid, x, y);
// 	else if (x > mid) return query(t[rt].r, mid+1, hi, x, y);
// 	return query(t[rt].l, lo, mid, x, mid)	+ query(t[rt].r, mid+1, hi, mid+1, y);
// }
// int main() {
// 	int i, n, q;
// 	cin >> n >> q;
// 	for(i = 0; i < n; i++) cin >> a[i+1];
// 	root[0] = build(0, MX);
// 	for(i = 1; i <= n; i++){
// 		root[i] = update(root[i-1], 0, MX, a[i]);
// 	}
// 	while(q--){
// 		int l, r, x, y;
// 		cin >> l >> r >> x >> y;
// 		cout << query(root[r], 0, MX, x, y) - query(root[l-1], 0, MX, x, y) << endl;
// 	}
// 	return 0;
// }







// tenplate for polciy based DS

#include <ext/pb_ds/assoc_container.hpp> // Common file 
using namespace __gnu_pbds; 
typedef tree<int, null_type, less<int>, rb_tree_tag, 
             tree_order_statistics_node_update> 
    new_data_set; 


typedef tree<pair<int,int>, null_type, less<pair<int,int>>, rb_tree_tag, 
             tree_order_statistics_node_update> 
    ordered_set; 





struct RollingHash{
	vector<ll> pwr, hsh;
	ll A, M;
	ll n;

    RollingHash(){}
    
	RollingHash(string s, ll _A = 31, ll _M = 1e9 + 7){
		n = s.size();
		pwr.resize(n+1); hsh.resize(n+1);

		A = _A, M = _M;

		pwr[0] = 1;
		for(ll i = 1; i <= n; i++) pwr[i] = pwr[i-1] * A % M;

		hsh[0] = s[0] % M + 1;
		for(ll i = 1; i < n; i++){
			hsh[i] = (hsh[i - 1] * A % M) + s[i] + 1; if(hsh[i] >= M) hsh[i] -= M;
		}
	}

	ll getHash(ll x, ll y){
		assert(x >= 0 and x < n and y >= 0 and y <= n);
		return (hsh[y] + M - ((x-1 >= 0)? hsh[x-1] * pwr[y-x+1] % M : 0)) % M;
	}
};

struct PalindromeChecker {
	RollingHash hash;
	RollingHash revHash;
    int n;

	PalindromeChecker(string s): hash(s), n(s.size()) {
        reverse(s.begin(), s.end());
        revHash = RollingHash(s);
    }

    bool isPalindrome(int i, int j) {
        return hash.getHash(i, j) == revHash.getHash(n-j-1, n-i-1);
    }

};

// COMBINATIONS

const int maxn = 109;
int C[maxn + 1][maxn + 1];

void COM() {
        C[0][0] = 1;
        for (int n = 1; n <= maxn; ++n) {
        C[n][0] = C[n][n] = 1;
        for (int k = 1; k < n; ++k)
                C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
        }
}

int n; // number of vertices
vector<vector<int>> adj; // adjacency list of graph
vector<bool> visited;
vector<int> ans;

void dfs(int v) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u])
            dfs(u);
    }
    ans.push_back(v);
}

void topological_sort() {
    visited.assign(n, false);
    ans.clear();
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs(i);
    }
    reverse(ans.begin(), ans.end());
}


// RMQ
class SegmentTree{
	long long n ;
	vector<long long>A,st;
	
	long long left(long long p)
	{
		return (p<<1);
	}
	
	long long right(long long p)
	{
		return (p<<1)+1;
	}
	
	void build(long long p,long long l,long long r)
	{
		if(l==r)
		st[p]=l;
		else
		{
			build(left(p),l,(l+r)/2);
			build(right(p),((l+r)/2)+1,r);
			long long li=st[left(p)];
			long long ri=st[right(p)];
			
			if(A[li]<A[ri])
			st[p]=li;
			else
			st[p]=ri;				
		}
	}
	
	long long rmq(long long p,long long l,long long r,long long i,long long j)
	{
		if(i>r || j<l)
		return -1;
		else if(l>=i && r<=j)
		return st[p];
		
		long long li=rmq(left(p),l,(l+r)/2,i,j);
		long long ri=rmq(right(p),((l+r)/2)+1,r,i,j);
		
		if(li==-1)
		return ri;
		else if(ri==-1)
		return li;
		else
		{
			if(A[li]<A[ri])
			return li;
			else
			return ri;
		}
	}
	
	public:
		SegmentTree(const vector<long long> &_A)
		{
			A=_A;
			n=(long long)(A.size());
			st.assign(4*n,0);
			build(1,0,n-1);
		}
		
		long long rmq(long long i,long long j)
		{
			return rmq(1,0,n-1,i,j);
		}
};







template<typename T>
class SegmentTree {
private: 
	int n;
	vector<T> tree;
	vector<T> lazy;
 
	int left(int pos) {
		return (pos << 1) + 1;
	}
 
	int right(int pos) {
		return (pos << 1) + 2;
	}
 
	int mid(int l, int r) {
		return ((l + r) >> 1);
	}
 
	T identity() {
		return 0;
	}
 
	T combine(T left, T right) {
		return left + right;
	}
 
	void lazyUpdate(int pos, int l, int r) {
		if(lazy[pos] == -1) return;
		tree[pos] = lazy[pos] * (r - l + 1);
		if(l != r) {
			lazy[left(pos)] = lazy[pos];
			lazy[right(pos)] = lazy[pos];
		}
		lazy[pos] = -1;
	}
 
	void build(int pos, int l, int r, vector<T> &arr) {
		if(l == r) {
			tree[pos] = arr[l];
		} else {
			build(left(pos), l, mid(l, r), arr);
			build(right(pos), mid(l, r) + 1, r, arr);
			tree[pos] = combine(tree[left(pos)], tree[right(pos)]);
		}
	}
 
	T query(int pos, int l, int r, int L, int R) {
		lazyUpdate(pos, l, r);
		if(r < L || l > R || l > r) {
			return identity();
		}
		if(L <= l && r <= R) {
			return tree[pos];
		}
		T q1 = query(left(pos), l, mid(l, r), L, R);
		T q2 = query(right(pos), mid(l, r) + 1, r, L, R);
		return combine(q1, q2);
	}
 
	void update(int pos, int l, int r, int L, int R, int val) {
		lazyUpdate(pos, l, r);
		if(r < L || l > R || l > r) {
			return;
		}
		if(L <= l && r <= R) {
			lazy[pos] = val;
			lazyUpdate(pos, l, r);
			return;
		}
		update(left(pos), l, mid(l, r), L, R, val);
		update(right(pos), mid(l, r) + 1, r, L, R, val);
		tree[pos] = combine(tree[left(pos)], tree[right(pos)]);
	}
 
public:
	SegmentTree(vector<T> arr) {
		n = arr.size();
		tree.assign(4 * n, 0);
		lazy.assign(4 * n, -1);
		build(0, 0, n - 1, arr);
	}
 
	T query(int l, int r) {
		return query(0, 0, n - 1, l, r);
	}
 
	void update(int l, int r, int val) {
		update(0, 0, n - 1, l, r, val);
	}
};
 

 // Trie
 class Trie {
    public:
    Trie *child[26];
    bool isend;
    Trie() {
        for(int i=0; i<26; i++) {
            child[i] = NULL;
        }
        isend = false;
    }

    void add(string s) {    
        int n = s.size();
        Trie *curr = this;
        for(int i=0; i<n; i++) {
            int v = s[i] - 'a';
            if(curr->child[v] == NULL) {
                curr->child[v] = new Trie();
            } 
            curr = curr -> child[v];
        }
        curr->isend  = true;
    }

    bool exists(string word) {
        Trie *curr = this;
        if(curr == NULL) {
            return false;
        }
        int n = sz(word);
        for(int i=0; i<n; i++) {
            int v = word[i] - 'a';
            if(curr->child[v] == NULL) {
                return false;
            }
            curr = curr -> child[v];
        }
        return curr->isend;
    }

    bool startswith(string p) {
        Trie *curr = this;
        if(curr == NULL) {
            return false;
        }
        int n = p.size();
        for(int i=0; i<n; i++) {
            int v = p[i] - 'a';
            if(curr->child[v] == NULL) {
                return false;
            }
            curr = curr -> child[v];
        }
        return true;
    }
    bool fun(string word, Trie *curr) {
        if(curr == NULL) {
            return false;
        }
        int n = word.size();
        bool ans = false;
        int i=0;
        for(i=0; i<n; i++) {
            if(word[i] == '.') {
                for(int j=0; j<26; j++) {
                    ans = ans || fun(word.substr(i+1), curr->child[j]);
                }
                return ans;
            }
            int v = word[i] - 'a';
            if(curr->child[v] == NULL) {
                return false;
            }
            curr = curr -> child[v];
        }

        return curr->isend;
    }
};

