
    ' ### [if] ### [then] marker ###
    
    dos title Tensorflow MNIST tutorial playground
    
    \ Common tools 
    
    

    marker ---xray---
        
    \ Initial Check 
    dos title working play ground
    cr version drop ." Current directory is : " cd
    dos if exist MNIST_data exit 34157
    34157 = [if] 
        ." TenserFlow dataset ./MNIST_data is existing. Good! Good! Good! let's go on ...." cr 
        \ exit \ <---------- exit to tutorial
    [else]
        ." TenserFlow dataset at ./MNIST_data is expected but not found!" cr
        ." Move it over to here if you have it already." cr
        \ ." Type <Enter> to proceed downloading it or 'abort' to STOP "
        \ accept char abort = [if] ." Action aborted by user." cr bye \ terminate
        \ [else] exit *debug* 22 ( <---------------- exit to tutorial ) [then] 
    [then]
    ." Type <Enter> to proceed " accept drop 
    py: sys.modules['pdb'].set_trace()
    stop
    
    \ ------------------- Never Land -------------------------------------------
    
    ." Error!! You reached to the never land, what's the problem?" cr
    ." Error!! You reached to the never land, what's the problem?" cr
    ." Error!! You reached to the never land, what's the problem?" cr
    ." Error!! You reached to the never land, what's the problem?" cr
    ." Error!! You reached to the never land, what's the problem?" cr
    ." Press enter to continue but don't!" accept

    \ 抽換 marker 界線，把 --- 改成 ---xray---
        <accept> <text> 
        locals().update(harry_port());  # bring in all FORTH value.outport
        dictate("-xray- marker ---xray---"); outport(locals()) # bring out all locals()
        </text> -indent py: exec(pop())
        </accept> dictate 
    
    \ This snippet adds batch_X, batch_Y into value.outport for investigation
        <accept> <text> 
        locals().update(harry_port());  # bring in all things
        # ------------ get what we want --------------------------
        batch_X, batch_Y = mnist.train.next_batch(100);  
        # ------------ get what we want --------------------------
        dictate("---xray--- marker ---xray---"); outport(locals()) # bring out all things
        </text> -indent py: exec(pop())
        </accept> dictate 

    bp11> batch_X :> [0].shape . cr
    (28, 28, 1)
    bp11> batch_X :> shape . cr
    (100, 28, 28, 1)
    bp11>
    bp11> batch_Y :> shape . cr
    (100, 10)
    bp11> batch_Y :> [0] . cr
    [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
    bp11>

    \ If we can see placeholders X and Y then we can see anything...
    <text>
    locals().update(harry_port());  # bring in all things
    myX = sess.run(X,feed_dict={X: batch_X, Y_: batch_Y})
    myY = sess.run(Y,feed_dict={X: batch_X, Y_: batch_Y})
    ok(cmd="---xray--- marker ---xray--- exit");  # clear things in forth
    outport(locals())
    </text> -indent py: exec(pop()) 

    \ it works!!! hahahaha
    bp11> words
    ... snip ....
    yclude) pyclude .members .source dos cd ### --- __name__ __doc__ __package__ 
    __loader__ __spec__ __annotations__ __builtins__ __file__ __cached__ peforth 
    tf tensorflowvisu mnist_data mnist X Y_ W b XX Y cross_entropy 
    correct_prediction accuracy train_step allweights allbiases I It datavis init 
    sess training_step batch_X batch_Y myX myY
    bp11>              ^^^^^^^^^^^^^^^^^^^^^^^^------Bingo!!                

    
    \ [x] exit doesn't work?    
        p e f o r t h    v1.09
        source code http://github.com/hcchengithub/peforth
        Type 'peforth.ok()' to enter forth interpreter, 'exit' to come back.

        Current directory is : c:\Users\hcche\Documents\GitHub\ML\tensorflow-mnist-tutorial
        TenserFlow dataset at ./MNIST_data is expected but not found!
        Move it over to here if you have it already.
        Type <Enter> to proceed downloading it or 'abort' to STOP
        Error!! You reached to the never land, what's the problem?
        Error!! You reached to the never land, what's the problem?
        Error!! You reached to the never land, what's the problem?
        Error!! You reached to the never land, what's the problem?
        Error!! You reached to the never land, what's the problem?
        Press enter to continue but don't!
        --> run setup.bat update new code that has fixed the probelm.

    [ ] mnist_1.0f_softmax.py stack is not empty at beginning
        bp11> .s
              0:           0           0h (<class 'int'>)
              1:           0           0h (<class 'int'>)
        bp11>    
    