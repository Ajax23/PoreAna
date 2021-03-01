################################################################################
# Geometry                                                                     #
#                                                                              #
"""Here basic geometric functions are noted."""
################################################################################


import math


def vector(pos_a, pos_b):
    """Calculate the vector between to two positions
    :math:`\\boldsymbol{a},\\boldsymbol{b}\\in\\mathbb{R}^n`

    .. math::

        \\text{vec}(\\boldsymbol{a},\\boldsymbol{b})
        =\\begin{pmatrix}b_1-a_1\\\\\\vdots\\\\b_n-a_n\\end{pmatrix}

    The two inputs can either be atom indices or to vectoral positions.

    Parameters
    ----------
    pos_a : integer, list
        First position :math:`\\boldsymbol{a}`
    pos_b : integer, list
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


def dotproduct(vec_a, vec_b):
    """Calculate the dotproduct of two vectors
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
    """Calculate the length of a vector :math:`\\boldsymbol{a}\\in\\mathbb{R}^n`

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
    return math.sqrt(dotproduct(vec, vec))


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
    angle = math.acos(dotproduct(vec_a, vec_b)/(length(vec_a)*length(vec_b)))

    return angle*180/math.pi if is_deg else angle
