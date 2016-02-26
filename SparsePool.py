class SparsePoolGrad(PoolGrad):
    def __init__(self, ds, ignore_border, st=None, padding=(0, 0)):
        PoolGrad.__init__(self, ds, ignore_border, st, padding, mode='max')

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        assert self.mode == 'max'
        x, maxout, gz = inp
        gx_stg, = out
        # number of pooling output rows
        pr = maxout.shape[-2]
        # number of pooling output cols
        pc = maxout.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        gx = numpy.zeros_like(y)
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = builtins.max(r * st0, self.padding[0])
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = builtins.max(c * st1, self.padding[1])
                        col_end = builtins.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y[n, k, row_ind, col_ind]):
                                    gx[n, k, row_ind, col_ind] += gz[n, k, r, c]
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, maxout, gz = inp
        ggx, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                DownsampleFactorMaxGradGrad(
                    self.ds, ignore_border=self.ignore_border,
                    st=self.st, padding=self.padding)(x, maxout, ggx)]

    def c_code(self, node, name, inp, out, sub):
        assert self.mode == 'max'
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        return """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        if ((x_typenum != z_typenum) || (x_typenum != gz_typenum))
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(z)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "z must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(gz)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a 4d ndarray");
            %(fail)s;
        }
        int z_r, z_c;
        z_r = PyArray_DIMS(%(z)s)[2];
        z_c = PyArray_DIMS(%(z)s)[3];
        int r, c; // shape of the padded_input
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        // allocating memory for gx
        if ((!%(gx)s)
          || !PyArray_ISCONTIGUOUS(%(gx)s)
          || *PyArray_DIMS(%(gx)s)!=4
          ||(PyArray_DIMS(%(gx)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(gx)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(gx)s)[2] != PyArray_DIMS(%(x)s)[2])
          ||(PyArray_DIMS(%(gx)s)[3] != PyArray_DIMS(%(x)s)[3])
          )
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(x)s), x_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(gx)s, 0);
        }
        int r_st, r_end, c_st, c_end; // used to index into the input img x
        dtype_%(z)s maximum; // temp var for maximum value in a region
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // the maximum value
                    maximum = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,i,j)))[0];
                    // the gradient corresponding to this maximum value in z
                    dtype_%(gz)s * gz = (
                          (dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s, b, k, i, j)));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        dtype_%(gx)s * gx = (
                          (dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s, b, k, m, n)));
                        if (a == maximum){
                          gx[0] = gx[0] + gz[0];
                        }
                      }
                    }
                  }
                }
              }
            }
        }
        """ % locals()

    def c_code_cache_version(self):
        return (0, 7)