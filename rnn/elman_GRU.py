import theano
import numpy
import os

from theano import tensor as T

# from collections import OrderedDict
from theano.compat.python2x import OrderedDict

dtype = theano.config.floatX
uniform = numpy.random.uniform
sigma = T.nnet.sigmoid
softmax = T.nnet.softmax
 
class model(object):

    def __init__(self, nh, nc, ne, de, cs, em, init, featdim):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model

        self.featdim = featdim

        tmp_emb = 0.2 * numpy.random.uniform(-1.0, 1.0, (ne+1, de))
        if init:
            for row in xrange(ne+1):
                if em[row] is not None:
                    tmp_emb[row] = em[row]

        self.emb = theano.shared(tmp_emb.astype(theano.config.floatX)) # add one for PADDING at the end

        # weights for LSTM
        n_in = de * cs
        print "de,cs", de, cs

        # n_hidden = n_i = n_c = n_o = n_f = nh
        n_hidden = n_z = n_r = nh
        n_y = nc
        print "n_y", n_y
        print "n_hidden, n_z, n_r ,nh", n_hidden, n_z, n_r ,nh

        self.W_xz = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_z)).astype(dtype))
        self.W_hz = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_z)).astype(dtype))
        self.b_z = theano.shared(numpy.cast[dtype](uniform(-0.5,.5,size = n_z)))
        
        self.W_xr = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_r)).astype(dtype))
        self.W_hr = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_r)).astype(dtype))
        self.b_r = theano.shared(numpy.zeros(n_r, dtype=dtype))
        
        self.W_hh = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_hidden)).astype(dtype))
        self.W_xh = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_hidden)).astype(dtype))
        self.b_h = theano.shared(numpy.cast[dtype](uniform(0.5, 1.5,size = n_hidden)))

        self.W_hy = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        self.b_y = theano.shared(numpy.zeros(n_y, dtype=dtype))

        self.h0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        # self.h0 = T.tanh(self.c0)
        
        # bundle weights
        self.params = [self.emb, self.W_xz, self.W_hz, self.b_z, self.W_xr,self.W_hr,  self.b_r, \
                        self.W_hh, self.W_xh, self.b_h,  self.W_hy, self.b_y, self.h0]
        self.names  = ['embeddings', 'W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', \
                        'W_hh', 'W_xh', 'b_h', 'W_hy', 'b_y', 'h0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        # print idxs.shape()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        # print type(x), x.shape(), "details of x"
        f = T.matrix('f')
        f.reshape( (idxs.shape[0], featdim))
        # print type(f), f.shape(), "details of f"
        y = T.iscalar('y') # label
        # print type(y), y.shape(), "details of y"

        def recurrence(x_t, feat_t, h_tm1):
            # i_t = sigma(theano.dot(x_t, self.W_xi) + theano.dot(h_tm1, self.W_hi) + theano.dot(c_tm1, self.W_ci) + self.b_i)
            # f_t = sigma(theano.dot(x_t, self.W_xf) + theano.dot(h_tm1, self.W_hf) + theano.dot(c_tm1, self.W_cf) + self.b_f)
            # c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, self.W_xc) + theano.dot(h_tm1, self.W_hc) + self.b_c)
            # o_t = sigma(theano.dot(x_t, self.W_xo)+ theano.dot(h_tm1, self.W_ho) + theano.dot(c_t, self.W_co)  + self.b_o)
            # h_t = o_t * T.tanh(c_t)

            z_t = sigma(theano.dot(x_t, self.W_xz) + theano.dot(h_tm1, self.W_hz) + self.b_z)
            ###### THIS IS DIFFERENT
            r_t = sigma(theano.dot(x_t, self.W_xr) + theano.dot(h_tm1, self.W_hr) + self.b_r)
            h_t = (T.tanh( theano.dot(x_t, self.W_xh)+ theano.dot(h_tm1*r_t, self.W_hh) + self.b_h)*(T.ones_like(z_t) - z_t)) + h_tm1*z_t
            # h_t = T.tanh(h_tm1)

            if self.featdim > 0:
                all_t = T.concatenate([h_t, feat_t])
            else:
                all_t = h_t
                
            # print "all_t", type(all_t), T.shape(all_t)
            s_t = softmax(theano.dot(all_t, self.W_hy) + self.b_y)
            # print T.shape(h_t), T.shape(c_t), T.shape(s_t)
            return [h_t,  s_t]

        # Initialization occurs in outputs_info
        # scan gives -- result, updates
        [h,  s], _ = theano.scan(fn=recurrence, sequences=[x,f], outputs_info=[self.h0,  None], n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs, f], outputs=y_pred)

        self.train = theano.function(inputs=[idxs, f, y, lr], outputs=nll, updates=updates)

        self.normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
