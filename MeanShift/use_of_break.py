#####
#THIS IS FOR MY OWN REFERENCE. THIS HAS NOT MUCH TO DO WITH THE MEAN SHIFT CLUSTERING ALGORITHM THAT I HAVE IMPLEMENTED.
#####

#the following logic has been used in the filtering of the cluster centers lying very clode to each other.
#this is used a rough idea of how we can use break intelligently to check for a condition and if the condition satisfies, we can start over the while loop.
#we will only get out of the while loop when the condition doesnt satisfy even once. 

while True:
brk = False
    for i in range(5):
        #for j in range(5):
            for k in range(5):
                print(i,k)
                if (i,k) == (3,1):
                    #a=False
                    brk = True
                    break #breaks out of the current for loop
            if brk == True:
                break
            print('here')
            print('-------')
    print('good')
    if not brk:
        break