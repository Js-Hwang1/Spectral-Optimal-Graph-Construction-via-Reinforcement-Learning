1. Does superior where k = n/2
2. Lambda2 falls after that superior case. (Possible to add edges to the superior structure to maintain the gap?)
3. Since k=n/2 only occurs when n is even 

1. On odd n; we loose from k = 3 to k = floor(n/2) for all odd n and begin to win just a little bit after ceil(n/2); We should address this behavior of the lower k behavior of odd n.

1. Add edges to strong graph families (k=3 mobius); and create a monotone lambda2 increase overall




_________________

Think fo the converse problem:
For Dense regime the complete-multipartie works really well (Has very high, near optimal algebraic connectivity). But why isn't there a special case for sparse case? 

Sparse: 
- Circulant is really bad...
- The compliemnt of the multipartite is really bad (as expected)

Try: 
- Take out the "super bad dense graph from complete", like Make dense circulant and take that out.?.

    test_cases = []
    for n in range(8,33):
        for k in range(3,n//2):
            test_cases.append( (n,k) )