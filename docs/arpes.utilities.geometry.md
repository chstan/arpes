# arpes.utilities.geometry module

**arpes.utilities.geometry.polyhedron\_intersect\_plane(poly\_faces,
plane\_normal, plane\_point, epsilon=1e-06)**

> Determines the intersection of a convex polyhedron intersecting a
> plane. The polyhedron faces should be given by a list of np.arrays,
> where each np.array at index *i* is the vertices of face *i*.
> 
> As an example, running \[p\[0\] for p in
> ase.dft.bz.bz\_vertices(np.linalg.inv(cell).T)\] should provide
> acceptable input for a unit cell *cell*.
> 
> The polyhedron should be convex because we construct the convex hull
> in order to order the points.
> 
>   - Parameters
>     
>       - **poly\_faces** –
>       - **plane\_normal** –
>       - **plane\_point** –
>       - **epsilon** –
> 
>   - Returns

**arpes.utilities.geometry.segment\_contains\_point(line\_a, line\_b,
point\_along\_line, check=False, epsilon=1e-06)**

> Determines whether a segment contains a point that also lies along the
> line. If asked to check, it will also return false if the point does
> not lie along the line.
> 
>   - Parameters
>     
>       - **line\_a** –
>       - **line\_b** –
>       - **point\_along\_line** –
>       - **check** –
>       - **epsilon** –
> 
>   - Returns

**arpes.utilities.geometry.point\_plane\_intersection(plane\_normal,
plane\_point, line\_a, line\_b, epsilon=1e-06)**

> Determines the intersection point of a plane defined by a point and a
> normal vector and the line defined by line\_a and line\_b. All should
> be numpy arrays.
> 
>   - Parameters
>     
>       - **plane\_normal** –
>       - **plane\_point** –
>       - **line\_a** –
>       - **line\_b** –
>       - **epsilon** –
> 
>   - Returns
