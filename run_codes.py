import runpy

        
    
for num in range(1, 15):
    print('start training' + '=' * 10, num)
    exp = 'exp_' + str(num)
    runpy.run_path(f'./{exp}/run.py')
    print('end training' + '=' * 10, num)
    
    print('start test' + '=' * 10, num)
    exp = 'exp_' + str(num)
    runpy.run_path(f'./{exp}/cross_test.py') 
    print('end test' + '=' * 10, num)