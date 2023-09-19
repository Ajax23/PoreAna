################################################################################
# Geometry                                                                     #
#                                                                              #
"""Here basic geometric functions are noted."""
################################################################################


import math


def dot_product(vec_a, vec_b):
    """Calculate the dot product of two vectors
    :math:`\\boldsymbol{a},\\boldsymbol{b}\\in\\mathbb{R}^n`

    .. math::

        \\text{dot}(\\boldsymbol{a},\\boldsymbol{b})=
        \\begin{pmatrix}a_1\\\\\\vdots\\\\a_n\\end{pmatrix}\\cdot
        \\begin{pmatrix}b_1\\\\\\vdots\\\\b_n\\end{pmatrix}=
        a_1\\cdot b_1+a_2\\cdot b_2+\\dots+a_n\\cdot b_n.

    Parameters
    ----------
    vec_a : list
        First vector :math:`\\boldsymbol{a}`
    vec_b : list
        Second vector :math:`\\boldsymbol{b}`

    Returns
    -------
    dot : float
        Dot product value
    """
    return sum((a*b) for a, b in zip(vec_a, vec_b))


def length(vec):
    """Calculate the length of a vector
    :math:`\\boldsymbol{a}\\in\\mathbb{R}^n`

    .. math::

        \\text{length}(\\boldsymbol{a})=|\\boldsymbol{a}|
        =\\sqrt{\\boldsymbol{a}\cdot\\boldsymbol{a}}

    Parameters
    ----------
    vec : list
        Vector a

    Returns
    -------
    length : float
        Vector length
    """
    return math.sqrt(dot_product(vec, vec))


def vector(pos_a, pos_b):
    """Calculate the vector between to two positions
    :math:`\\boldsymbol{a},\\boldsymbol{b}\\in\\mathbb{R}^n`

    .. math::

        \\text{vec}(\\boldsymbol{a},\\boldsymbol{b})
        =\\begin{pmatrix}b_1-a_1\\\\\\vdots\\\\b_n-a_n\\end{pmatrix}

    Parameters
    ----------
    pos_a : list
        First position :math:`\\boldsymbol{a}`
    pos_b : list
        Second position :math:`\\boldsymbol{b}`

    Returns
    -------
    vector : list
        Bond vector
    """
    # Check dimensions
    if not len(pos_a) == len(pos_b):
        print("Vector: Wrong dimensions...")
        return

    # Calculate vector
    return [pos_b[i]-pos_a[i] for i in range(len(pos_a))]


def unit(vec):
    """Transform a vector :math:`\\boldsymbol{a}\\in\\mathbb{R}^n` into a
    unit vector

    .. math::

        \\text{unit}(\\boldsymbol{a})
        =\\frac{\\boldsymbol{a}}{|\\boldsymbol{a}|}

    Parameters
    ----------
    vec : list
        Vector a

    Returns
    -------
    vec : list
        Vector
    """
    vec_length = length(vec)

    return [x/vec_length if not vec_length == 0 else x for x in vec]


def cross_product(vec_a, vec_b):
    """Calculate the cross product of two three-dimensional vectors
    :math:`\\boldsymbol{a},\\boldsymbol{b}\\in\\mathbb{R}^3`

    .. math::

        \\text{cross}(\\boldsymbol{a},\\boldsymbol{b})=\\begin{pmatrix}
        a_2\\cdot b_3-a_3\\cdot b_2\\\\
        a_3\\cdot b_1-a_1\\cdot b_4\\\\
        a_1\\cdot b_2-a_2\\cdot b_1
        \\end{pmatrix}

    Parameters
    ----------
    vec_a : list
        First vector :math:`\\boldsymbol{a}`
    vec_b : list
        Second vector :math:`\\boldsymbol{b}`

    Returns
    -------
    vec : list
        Cross product vector
    """
    vec = []
    vec.append(vec_a[1]*vec_b[2]-vec_a[2]*vec_b[1])
    vec.append(vec_a[2]*vec_b[0]-vec_a[0]*vec_b[2])
    vec.append(vec_a[0]*vec_b[1]-vec_a[1]*vec_b[0])

    return vec


def angle(vec_a, vec_b, is_deg=True):
    """Calculate the angle between two vectors
    :math:`\\boldsymbol{a},\\boldsymbol{b}\\in\\mathbb{R}^n`

    .. math::

        \\text{angle}=\\cos^{-1}\\frac{\\boldsymbol{a}\cdot\\boldsymbol{b}}
        {|\\boldsymbol{a}||\\boldsymbol{a}|}

    Parameters
    ----------
    vec_a : list
        First vector :math:`\\boldsymbol{a}`
    vec_b : list
        Second vector :math:`\\boldsymbol{b}`
    is_deg : bool, optional
        True if the output should be in degree

    Returns
    -------
    angle : float
        Angle
    """
    angle = math.acos(dot_product(vec_a, vec_b)/(length(vec_a)*length(vec_b)))

    return angle*180/math.pi if is_deg else angle
