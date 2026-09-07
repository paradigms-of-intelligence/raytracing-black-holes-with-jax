
# Coordinate Transformations, "Fictitious Forces", Christoffel Symbols, and All That

These notes should - ideally - provide some end-to-end understandable
write-up of "accelerations observed in accelerating
(i.e. non-inertial) coordinate systems.

The motivation for this write-up came from a desire to get a better
grip on questions such as "how does one satellite in a constellation
accelerate towards or away from another satellite" (where both "are
not inertial in a Newtonian-Physics approach due to feeling the effect
of gravity"), or, more subtly, "what is the Coriolis force
contribution that's attributable to our orbital plane precessing in a
sun-synchronous orbit?".

The derivations here are intentionally verbose, the idea being that at
every point we should be able to relate intermediate quantities to the
"velocities" and "accelerations" attributed to moving objects in
different coordinate systems. At a purely algebraic level, the
calculation could be shortened to less than one page by leaning more
heavily into the chain rule, but that then would require more of an
expert's intuition to see how the expressions align with physical
notions. So, rather than just doing Taylor expansion of the motion of
an accelerating object subjected to a nonlinear coordinate
transformation, we spend all the effort required to see the first
nonlinear terms that affect observed accelerations for coordinate
systems related by some nonlinear transformation. We will keep all
summations explicit, and only take small and simple steps in algebraic
manipulations.

This should - hopefully - keep explanations accessible to readers who
have learned how to do some multivariate calculus, but do not have
years of hands on experience with that kind of mathematics.

## Some Basic Maths

We want to talk about coordinate transformations. In general, we will
use $(t, x, y, z)$-coordinates: Even if we do not work with space-time
in the relativistic sense, it simplifies the discussion a lot if we
can describe "events" as "this time and that position"
coordinate-4-tuples.

Let us consider having two different "coordinate systems", i.e. two
different ways to attribute a 4-tuple of numbers to some given event
$E$. Let's use capital letters for "System A"-coordinates and small
letters for "System B" coordinates. Loosely:

$E@A = (T,X,Y,Z),\quad E@B = (t,x,y,z)$

In general, we will have functions that express $TXYZ$-coordinates in
terms of $txyz$-coordinates and vice-versa, and whenever we need to
look at local coordinate-transformations at a point, we can think of
the relevant functions to be described using either the
$TXYZ$-coordinates or $txyz$-coordinates. In many situations, one
choice is more convenient, and it pays to think a little about what
approach works better. We will come to this.

Let's consider a simple situation: We are given a trajectory of an
object (such as: a satellite) in terms of a 4-coordinate-tuple valued
function of some curve-parameter $\tau$, which we here can take to be
time. So, for position-coordinates in the $A$-coordinate-system, we
have: $P@A(\tau)=(T(\tau), X(\tau), Y(\tau), Z(\tau))$,
i.e. $P^0(\tau)@A=T(\tau),\; P^1(\tau)=X(\tau)$ etc.

Two notes:

* We start index counting at zero, and index-zero is our
  time-coordinate.  This is aligned with common conventions in
  relativity, and makes our explorations carry over directly from
  Newtonian to Einsteinian gravitational physics.

  This is nothing to be scared about - rather, we should think of this
  as "let's align with the highly streamlined and systematic notation
  and approach used by people who have to analyze way hairier problems
  than everything we are going to encounter" rather than "let's stick
  with the idea to not use any maths beyond
  late-19th-century-engineering mathematics as much as possible". *We
  wouldn't want to do this with Roman Numerals either*.

* One perhaps slightly strange-looking consequence here is that we
  have a curve-parameter (above: $\tau$) that we can take to be simply
  "wall clock time". This gives us, for the time-coordinate of any
  coordinate system we want to use: $T(\tau)=t(\tau)=\tau$. Could we
  not then simplify all this by dropping this 4th coordinate "time"
  and staying 3d? At the conceptual level, the situation closely
  parallels the one with 3d computer graphics that is generally done
  with 4d projective coordinates: using the 4d approach simplifies the
  structure of the mathematics a lot, since we get "time/space
  coordinate mixing contributions" that otherwise would make
  conceptually very simple expressions have additional summands.

### Multi-dimensional Taylor Expansion

Let's say we have some analytic function of one variable, such as
perhaps $f(x)=\exp(-x^2/2)$. "Being analytic" means that if we expand
the function into a Taylor series at a given point $x_0$, then at
every point $x$ that is less than the radius of convergence away from
$x_0$, that series will converge to the original function's value. It
is by no means true that all functions are analytic, i.e. that there
is some neighborhood of $x_0$ inside which the Taylor series always
converges to the value of the original function, but for the problems
we are looking into, we can nevertheless take this to always hold.

We know we have - using somewhat informal writing ("you see how the
full sum will look like"):

$$f(x_0+h)=f(x_0)+(h^1/1!)f'(x_0)+(h^2/2!)f''(x_0)+(h^3/3!)f'''(x_0)+\ldots$$

**The good news**: "Physics" (mostly) makes us care only about
positions, velocities, and accelerations, not higher
derivatives. (Sometimes, the 3rd derivative, the "jerk" - or
rate-of-change of force, respectively, acceleration - will play a
role. Sudden onset/stopping of a force, such as when releasing a ball
from a sling, would be "infinite jerk".)

*So, we will here be able to truncate such expansions at the second
derivative*.

**The bad news**: We right away have to ponder how the above expansion
generalizes from $\mathbb{R}^1\to\mathbb{R}^1$ functions to
$\mathbb{R}^n\to\mathbb{R}^n$ functions - here specifically with
$n=4$.

More concretely, let's say we have a way to express the $T,X,Y,Z$
coordinates in terms of $t,x,y,z$-coordinates. The information-content
of such a $\mathbb{R}^4\to\mathbb{R}^4$ function is equivalent to
having a vector of four $\mathbb{R}^4\to\mathbb{R}^1$ functions, one
per coordinate.

So, if for example, $(t,x,y,z)$ are coordinates for an "inertial"
(not-accelerating, "free fall") coordinate system where objects not
experiencing any force will move with constant velocity, and the
$(T,X,Y,Z)$ coordinate system shares the same origin, but rotates
around the z-axis with angular velocity $\omega$, we could describe
this as:

$$\begin{array}{lcl}
\tilde T(t,x,y,z) &=& t\\
\tilde X(t,x,y,z) &=& x\cos(\omega t) - y\sin(\omega t)\\
\tilde Y(t,x,y,z) &=& x\sin(\omega t) + y\cos(\omega t)\\
\tilde Z(t,x,y,z) &=& z
\end{array}$$

Already here, we are seeing why it makes sense to treat $t$ as a
coordinate on the same footing. We are using the "tilde" above the
name of each coordinate to indicate that these are not coordinates
(numbers) but functions - but this is not some widely established
convention.

Let's use the name $X^i$ for the entries of the vector-of-functions
$(\tilde T, \tilde X, \tilde Y, \tilde Z)$. A few things are somewhat
unfortunate here:

* While it is customary in this context to place vector-coordinate
  indices "upstairs", this can cause some ambiguity
  w.r.t. exponent-or-index. In our explorations here, we will never
  encounter an exponent on a coordinate, resolving this question.

* We have a name clash $X^i$ (the collection of four
  coordinate-functions) and the coordinate $X=X^1$.

* There is a natural isomorphism between $p\mapsto(\tilde T(p), \tilde
  X(p), \tilde Y(p), \tilde Z(p))$, which "has type `R^4->R^4`" and
  $X^i$ as introduced above, where the function-tuple "has type
  `(R^4->R, R^4->R, R^4->R, R^4->R)`", and going forward, we will just
  liberally switch between these different pictures without saying so.

The latter is a source of potential confusion that makes learning
these things unnecessarily hard, and it totally is a consequence of a
culturally widely established lack of effort to be precise in
constructions.

So, how does the multi-dimensional Taylor expansion of the
coordinate-transformation function look like? As we said, we only need
to go to 2nd order.

$$X^i(p+h)=
X^i(p)
+(1/1!)\sum_m\frac{\partial X^i}{\partial x^m}h^m
+(1/2!)\sum_{m,n}\frac{\partial^2 X^i}{\partial x^m\partial x^n}h^m h^n
+\{\mbox{higher order terms}\}
$$

Here, $\frac{\partial X^i}{\partial x^m}$ is a matrix of partial
derivatives, "taking the derivative of every capital-letter
coordinate-function by every small-letter coordinate" which then gets
evaluated at the point with small-letter coordinates $p$ (this is
silent/implicit). Likewise for the quadratic term. So, if $Y(t,x,y,z)
= X^2(t,x,y,z) = y - 7x^3z^5 -2tx$, then $\frac{\partial Y}{\partial
x\,\partial z} = \frac{\partial X^2}{\partial x^1\partial
x^3}=105\,x^2z^4$.

Above, it is noteworthy that the quadratic term that has a 2nd partial
derivative which is a 3-index object: the totality of these
derivatives is indexed by $(i,m,n)$. This is not a "matrix", but
(equivalent to) a 3-index-array of coefficients - perhaps best thought
of not as some sort of 3d coordinate scheme, but a relational database
table with one "value" column and three index-columns. Here, we
already see how multiindex objects arise very naturally and "simple
matrix/vector linear algebra doesn't quite cut it".

### Body Motion

Our next step is about considering a moving body. Let's make
$\tau=t-t_0$ "time measured from some given reference point in
time". We only care about position, velocity, and acceleration, so the
momentary situation around $\tau=0$ can be described as follows:

$$
x^i(\tau)=\hat x^i + \hat v^i\tau + \frac{1}{2}\hat a^i\tau^2
+\{\mbox{higher order terms we do not care about}\}
$$

Here, $(\hat x^i, \hat v^i, \hat a^i)$ are the position-coordinates,
velocity-coordinates, and acceleration-coordinates for the momentary
position, velocity, and acceleration at $\tau=0$, expressed in the
(small-letters) $B$-coordinate-system. Clearly, this again is simply a
Taylor expansion which we truncated to 2nd order.

The next step is pretty clear now: We need to substitute this "object
motion to 2nd order in time" into the coordinate-transformation that
gets us $X^i$-coordinates in terms of $x^m$-coordinates, which we also
expand to 2nd order in coordinates. Since we are only interested in
accelerations in the $X^i$-coordinates, we can straight away drop all
terms $\propto \tau^3, \propto \tau^4, \ldots$.

Let us consider $x^i(\tau)=\hat x^i+h^i_\tau$, so $h^i_\tau = \hat
v^i\tau + \frac{1}{2} \hat a^i\tau^2$ and substitute into what we had
above:


$$X^i(p+h)=
X^i(p)
+\sum_m\frac{\partial X^i}{\partial x^m}h_\tau^m
+\frac{1}{2}\sum_{m,n}\frac{\partial^2 X^i}{\partial x^m\partial x^n}h^m_\tau h^n_\tau
+\{\ldots\}
$$

Expanding:

$$\begin{array}{lcl}
X^i(p_\tau)=X^i(p+h_\tau)&=&\\
&&X^i(p)\\
&&+\sum_m\frac{\partial X^i}{\partial x^m} \left(\hat v^m\tau + \frac{1}{2} \hat a^m\tau^2\right)\\
&&+\frac{1}{2}\sum_{m,n}\frac{\partial^2 X^i}{\partial x^m\partial x^n}
\left(\hat v^m\tau + \frac{1}{2} \hat a^m\tau^2\right)
\left(\hat v^n\tau + \frac{1}{2} \hat a^n\tau^2\right)
+\{\ldots\}
\end{array}
$$

Now, we note that from the quadratic term, we can only get a
$\propto\tau^2$-term if a $\hat v^m$-coordinate multiplies a $\hat
v^n$-coordinate. Everything else is $\propto \tau^3, \propto
\tau^4$. Collecting contributions by power-of-$\tau$, we get:

$$X^i(p_\tau)=X^i(p+h_\tau)=
X^i(p)
+\tau\cdot\sum_m\frac{\partial X^i}{\partial x^m}\hat v^m
+\frac{1}{2} \tau^2\cdot
\left(\sum_m \frac{\partial X^i}{\partial x^m}\hat a^m
+\sum_{m,n}\frac{\partial^2 X^i}{\partial x^m\partial x^n}
\hat v^m\hat v^n\right)
+\{\ldots\}\qquad(*)
$$

This already looks interesting:

1. The position in new coordinates we get by simply using the
   coordinate-transformation functions on the position in old
   coordinates.

2. The velocity in new coordinates is simply the velocity in old
   coordinates, transformed with the coordinate-transformation matrix.

3. The acceleration-vector in new coordinates is simply the
   velocity-vector in old coordinates, coordinate-transformed to the
   new coordinate system with the "how much change in the new 2nd
   coordinate per change in the old 3rd coordinate" type weighting
   "that relates $X^i$-change to $x^m$-change", *plus a new term that
   somehow involves "the bending of the $X$-coordinates expressed in
   the $x$-coordinates"*.

We will look into especially (3.) in more detail in a moment, but it
is worthwhile to ponder (2.) a bit: Let's consider the very simple
situation where $x^i$-coordinates describe coordinates "relative to
spaceship in interstellar space, no thrust, inertial motion", and
$X^i$ describe "coordinates relative to a probe released by the
spaceship. Same orientation of the coordinates, but
$X^i=x^i-b^it=x^i-bx^0$". Commonly, we would want to think about
velocities as satisfying $V^i=v^i-b^i$. Note that there are *two
summands* here, while in our above treatment, we only have a *single*
matrix-vector product! Indeed, if we were to work out how this matrix
looks like, we would get *exactly* the same situation that we know
from 3-d computer graphics in projective coordinates (as alluded to
earlier): The $\partial T/\partial t=1$-entry makes it possible to
absorb the shift into matrix multiplication!

### The relation to Christoffel Symbols

Above, we found how to express the $X^i$-coordinate-acceleration in
terms of coordinate-transformed $x^i$-coordinate-acceleration (as we
would expect), plus "some other term that depends on velocities and
gives us accelerations if the relation between old and new coordinates
is not linear".

Overall, this expression is pretty useful. For example, if we know the
acceleration of one satellite relative to another satellite in
Cartesian coordinates, but want to know the angle-acceleration (since
this will translate to some motor force) for a telescope whose
pointing direction we would like to describe in satellite-fixed
(latitude, longitude) coordinates, we would simply use this formula to
convert to (time, latitude, longitude, distance)-coordinates and could
immediately read off angle-accelerations.

#### Physics: Centrifugal and Coriolis Forces

In many applications, both coordinate systems will be such that the
spatial directions are Cartesian, but "these Cartesian coordinate
systems can move and perhaps rotate in time". Then, the
"velocity-transformation" gets us just the $\vec v\mapsto M\vec v+\vec
w$ rotation-plus-shift transformation we see in a "purely 3d"
treatment.

How about the "acceleration" terms? Here, it is noteworthy that, since
we always have $v^0=1$ in our approach, we get $a^0=0$, and hence
there is no such "shift"-term for the time-coordinate akin to the
shift-terms for spatial coordinates provided by the vector $\vec w$
above for the velocity-transformation. For the 2nd term, the one with
the "second partial derivative", if the relation between spatial
coordinates is linear-with-perhaps-an-offset (i.e. "affine"), *as we
have it if spatial coordinates at any time are Cartesian* (where we
then only are free to rotate and shift, for the given point in time),
if both derivatives are w.r.t. spatial coordinates, the corresponding
$\partial^2/\partial\partial$-coefficient is zero.

We hence only get "interesting forces" if one or both the partial
derivatives involve the time-coordinate.

Let's briefly specialize to a rotating coordinate system where time
only shows up in the coordinate-transformation formulae via
$\sin(\omega t)$ and $\cos(\omega t)$-factors. In that case, a
nontrivial first time-derivative will come with one power of $\omega$,
and a second time-derivative will come with a factor $\omega^2$. Note
that for centrifugal acceleration, we have $a=r\omega^2$. Indeed, this
"double time derivative" contribution from this term gives us the
centrifugal force.

What about the "mixed" term with one spatial and one time-derivative?
This will give us an acceleration-contribution that is proportional to
$\omega$, and also to the spatial part of $\hat v^m$ (the other
$v^?$-factor will here be $v^0=1$). We also get a factor-2 from
"either $v$-factor can be the $v^0$. This is the Coriolis force: In a
rotating system, an object moving at velocity $\vec v$ will experience
a "fictitious force" acceleration $2\vec v\times\vec\omega$, where
$\vec\omega$ is the rotation-vector (pointing along the axis of
rotation, length being the angular velocity.)

It makes sense to try to link this to intuition. Considering a tall
tower at the equator, if we drop a rock from the top of the tower, it
accelerates towards the ground. Will it land to the east or to the
west of the point right underneath where it was dropped (whose
position we could determine with a plumb line)? Pondering the
situation in a non-rotating coordinate system, the rock at the top of
the tower is further away from earth's rotation axis than the bottom
of the tower, so it is moving faster eastward than the bottom. As it
falls, it will retain that excess eastward velocity, which will make
it land further to the east. The "velocity excess" is directly
proportional to the vertical distance fallen, and seen as the eastward
velocity by a ground-fixed rotating observer. Distance fallen is
proportional to the square of the time passed since the rock was
dropped, so the *rate-of-change* of eastward velocity is proportional
to the time that has passed since dropping the rock - but so is
downward velocity. So, the most relevant contribution to eastward
acceleration (very small) comes from downward velocity, and -
unsurprisingly - also is proportional to earth's rotation rate,
$|v_{\text east}|\propto|\omega|\cdot|v_{\text{down}}|$. Also, the
right hand rule shows that the cross product of a "downward" vector
with the rotation-axis vector (towards the North Star) indeed points
eastward, in alignment with $\vec a_{\text{Coriolis}}=2\,\vec v \times
\vec\omega$.

### Finally: The Connection Coefficients

Let us now consider the most general situation. As we just have seen,
changing to a different coordinate system can give rise to
velocity-dependent accelerations.

In the generic case, *the "original" $B$-coordinate system already
comes with such velocity-dependent accelerations*, and it is natural
to expect that we simply have to coordinate-transform these to get
**a** contribution to the (coordinate-)acceleration in the "new"
coordinate system - but there will be another contribution that is
related to the "new" coordinates "accelerating relative to the 'old'
coordinates".

We hence want to consider not one, but a collection of different
trajectories in the $B$-coordinate system, and we describe these in
terms of accelerations that we understand as being a sum of two
contributions,

$$\hat a^m=\alpha^m - \sum_{n,p}\gamma^m{}_{np}\hat v^n \hat v^p\qquad({*}{*})$$

where $\hat v^n$ is the velocity in $B$-coordinates. So, the idea here
is that an observer who describes the world in the
$B$-coordinate-system, which in general can be a "non-inertial"
coordinate system, might regard an acceleration they observe as the
sum of some acceleration that an inertial observer would see (in that
observer's $C$-coordinate-system), but coordinate-transformed to their
own coordinate system - that would be the $\alpha^m$ term above - plus
some velocity-dependent acceleration-contributions that encode (for
example) the Coriolis force. So, from the perspective of an observer
using $B$-coordinates, if two bodies have *the same* acceleration in
$B$-coordinates $\hat a^m$, but one is at rest while the other one
moves with some $B$-coordinates velocity, then they would have
different accelerations in $C$-coordinates.

Now, what would the acceleration in $A$-coordinates then be? We can
simply read off that acceleration $A^i$ from the term multiplying
$\frac{1}{2}\tau^2$ in Eq. $(*)$:

$$
A^i=\sum_m \frac{\partial X^i}{\partial x^m}\hat a^m
+\sum_{m,n}\frac{\partial^2 X^i}{\partial x^m\partial x^n}
\hat v^m\hat v^n
$$

Now, let us substitute in the $\hat a^m$ from above:

$$
A^i=\sum_m \frac{\partial X^i}{\partial x^m}
\left(\alpha^m - \sum_{n,p}\gamma^m{}_{np}\hat v^n \hat v^p\right)
+\sum_{m,n}\frac{\partial^2 X^i}{\partial x^m\partial x^n}
\hat v^m\hat v^n
$$

Rearranging and relabeling summation indices:

$$
A^i=\sum_m \frac{\partial X^i}{\partial x^m}\alpha^m
+ \sum_{n,p}\left(
  \sum_m
  -\frac{\partial X^i}{\partial x^m}
  \gamma^m{}_{np}\hat v^n \hat v^p
+\frac{\partial^2 X^i}{\partial x^n\partial x^p}
\hat v^n\hat v^p\right)\qquad({*}{*}{*})
$$

What is noteworthy here is that Eq. $({*}{*})$ expressed the
velocity-dependent acceleration-corrections in $B$-coordinates that
get added to the
$C$-coordinates-accelerations-transformed-to-$B$-coordinates
$\alpha^m$ in terms of the $B$-coordinates velocities, while in the
expression above, we do almost the same thing, but the velocities we
are referring to are $B$-coordinate velocities. We would like these to
be $A$-coordinate velocities instead. We know how these velocities are
related - the $A$-velocities are the term multiplying $\tau$ in
Eq. $(*)$:

$$
V^i=\sum_m\frac{\partial X^i}{\partial x^m}\hat v^m.
$$

Since $\left(\frac{\partial X}{\partial x}\right)^{-1}=\left(\frac{\partial x}{\partial X}\right)$, or, more precisely,
$$
\sum_m\left(\frac{\partial X^i}{\partial x^m}\right)
\left(\frac{\partial x^m}{\partial X^j}\right)=\delta^i_j,
$$
we have:
$$
\hat v^m=\sum_i\frac{\partial x^m}{\partial X^i} V^i
$$

So, substituting this velocity-expression into $({*}{*}{*})$ gets us:

$$
A^i=\sum_m \frac{\partial X^i}{\partial x^m}\alpha^m
+ \sum_{n,p}\left(
  -\sum_m \frac{\partial X^i}{\partial x^m}
  \gamma^m{}_{np}
+\frac{\partial^2 X^i}{\partial x^n\partial x^p}\right)
\left(\sum_j \frac{\partial x^n}{\partial X^j}V^j\right)
\left(\sum_k \frac{\partial x^p}{\partial X^k}V^k\right)
$$

Let us rearrange and simplify a bit:

$$
A^i=\sum_m \frac{\partial X^i}{\partial x^m}\alpha^m
+ \sum_{m,n,p}\sum_{j,k}
  -\left(\gamma^m{}_{np}
  \frac{\partial X^i}{\partial x^m}
  \frac{\partial x^n}{\partial X^j}
  \frac{\partial x^p}{\partial X^k}\right)V^jV^k
+ \sum_{n,p}\sum_{j,k}
+\frac{\partial^2 X^i}{\partial x^n\partial x^p}
  \frac{\partial x^n}{\partial X^j}
  \frac{\partial x^p}{\partial X^k}V^jV^k
$$

Remember that in $B$-coordinates, we had, for velocity-dependent
acceleration terms (Eq. $({*}{*})$),

$$\hat a^m=\alpha^m - \sum_{n,p}\gamma^m{}_{np}\hat v^n \hat v^p$$

and so if we likewise want this to hold in $A$-coordinates in the form:

$$\hat A^i=\sum_m a^m\frac{\partial X^i}{\partial x^m} - \sum_{j,k}\Gamma^i{}_{jk}V^j V^k,$$

then comparing these expressions tells us that we want this to hold:

$$
-\Gamma^i{}_{jk} = \sum_{m,n,p}
  \left(-\gamma^m{}_{np}
  \frac{\partial X^i}{\partial x^m}
  \frac{\partial x^n}{\partial X^j}
  \frac{\partial x^p}{\partial X^k}\right)
+ \sum_{n,p}\frac{\partial^2 X^i}{\partial x^n\partial x^p}
  \frac{\partial x^n}{\partial X^j}
  \frac{\partial x^p}{\partial X^k}
$$

Let us rearrange this by multiplying with the appropriate inverse
matrices to extract the $\gamma^m{}_{np}$. (This might look a bit
scary, but if we think about it, it is a straightforward step: For
every "open index" that occurs at the left hand side, we
coordinate-transform with the "value of a small linear coordinate
update in one system expressed in the currency of the other system's
small linear coordinate updates" matrix. Three indices require three
such matrices. The formula looks complicated since we are looking at a
"wiring diagram" that should be seen as a graph but was rendered as a
mathematical term via typography instead. This is a bit like trying to
describe a reaction in organic chemistry by not showing molecules, but
their names only.)


$$
\begin{array}{l}
\sum_{m,n,p,i,j,k}
\gamma^m{}_{np}
\left(
  \frac{\partial X^i}{\partial x^m}
  \frac{\partial x^n}{\partial X^j}
  \frac{\partial x^p}{\partial X^k}
\right)
\left(
  \frac{\partial x^{m'}}{\partial X^i}
  \frac{\partial X^j}{\partial x^{n'}}
  \frac{\partial X^k}{\partial x^{p'}}
\right)
=\\
=
\sum_{i,j,k}
\Gamma^i{}_{jk}
\left(
  \frac{\partial x^{m'}}{\partial X^i}
  \frac{\partial X^j}{\partial x^{n'}}
  \frac{\partial X^k}{\partial x^{p'}}
\right)
+
\sum_{i,j,k}
\sum_{n,p}\frac{\partial^2 X^i}{\partial x^n\partial x^p}
  \frac{\partial x^n}{\partial X^j}
  \frac{\partial x^p}{\partial X^k}
\left(
  \frac{\partial x^{m'}}{\partial X^i}
  \frac{\partial X^j}{\partial x^{n'}}
  \frac{\partial X^k}{\partial x^{p'}}
\right)
\end{array}
$$

This leaves us with:

$$
\gamma^{m'}{}_{n'p'}
=
\sum_{i,j,k}
\Gamma^i{}_{jk}
\left(
  \frac{\partial x^{m'}}{\partial X^i}
  \frac{\partial X^j}{\partial x^{n'}}
  \frac{\partial X^k}{\partial x^{p'}}
\right)
+
\sum_{i}
\frac{\partial^2 X^i}{\partial x^{n'}\partial x^{p'}}
\frac{\partial x^{m'}}{\partial X^i}
$$

Or, after renaming indices:

$$
\gamma^{m}{}_{np}
=
\sum_{i,j,k}
\Gamma^i{}_{jk}
\left(
  \frac{\partial x^m}{\partial X^i}
  \frac{\partial X^j}{\partial x^n}
  \frac{\partial X^k}{\partial x^p}
\right)
+
\sum_{i}
\frac{\partial^2 X^i}{\partial x^n\partial x^p}
\frac{\partial x^m}{\partial X^i}.
$$

This is the known expression for expressing the "velocity-dependent
accelerations-corrections" of one coordinate system (here the
$B$-coordinate system) in terms of that of another coordinate system.

This then is the "transformation law for Christoffel Symbols (of the
2nd kind)". If we did not include the 2nd summand - the one with the
2nd partial derivatives - this would correspond to
coordinate-transforming the "parallel-transport connection for
in-general-not-Cartesian coordinate system A" so that we get the
"parallel-transport connection for coordinate system A, expressed in
the coordinates of coordinate system B". The 2nd term turns that
quantity into the "parallel-transport connection for coordinate system
B, expressed in the coordinates of coordinate system B", or shorter,
"connection for coordinate system B". Notably, if the
"parallel-transport connection for coordinate system A" is nonzero, it
will always be nonzero irrespective of what coordinate system we use
to describe it (since the coordinate transformations are
invertible). The 2nd, "inhomogeneous" term can make them zero for one
coordinate system while they are nonzero for another. Indeed, it is
always possible to locally make the $\Gamma^i{}_{jk}$ zero by
switching to the coordinate system of some (non-rotating) free-falling
observer. This gives us "Riemann Normal Coordinates".

