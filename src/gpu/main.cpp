// run in CPU

#include "process.h"

int main(int argc, char *argv[]){
    // cout << cuda::getCudaEnabledDeviceCount();

    // cuda::setDevice(0);

    if (argc <3 ) {
        cout<<"Lack of argument"<<endl;
    } else {
        try {
            if (argc == 3){
                string file = argv[1];
                Process xuly(file);
                xuly.ProcessingJSON(argv[2], false);
            }
            else {
                string data=argv[1];
                string file=argv[2];

                Process xuly(data, file);
                if (argc == 5){
                    xuly.ProcessingHomography(argv[3], argv[4]);
                }
                else{
                    xuly.ProcessingHomography(argv[3]);
                }
            }  
        } catch (Exception e){
            //cout << "false   " << "message: " << e.what() << endl;
        }     
    }

    // cuda::resetDevice();
    return 0;
}
