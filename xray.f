

    ### marker ###
    
    \ Initial and common tools 
    
    cr version drop 
    dos if exist MNIST_data exit 34157
    34157 = [if] 
        cr 
        ." Current directory is :" cr ."     " cd
        cr 
        ." TenserFlow dataset at ./MNIST_data is existing." cr
        ." Good! let's go on ...." cr exit
    [else]
        cr 
        ." Current directory is :" cr ."     " cd
        cr 
        ." TenserFlow dataset at ./MNIST_data is expected but not found!" cr
        ." Move it over to here if you have it already." cr
        ." Type 'abort' to STOP or <Enter> to proceed downloading it."
        py> input()=="abort" [if] ." Action aborted by user." cr abort [then] 
    [then]

    marker --- 
    
    \ Breakpoint temperary 
    
    
