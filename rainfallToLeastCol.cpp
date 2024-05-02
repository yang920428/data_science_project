#include<iostream>
#include<fstream>

using namespace std;

void write(string name,string num){
    string commonPath = "D://code_sets//ds_bigproject//data_set//txtdata//";
    string path_rawfile = commonPath + name + "_" + num + ".txt";
    string path_write = commonPath + "LSTM_input_data//" + name + ".txt";

    ifstream read;
    read.open(path_rawfile , ios::in);

    ofstream write;
    write.open(path_write , ios::out);

    float press , tempra , Td , RH , Precp , hour , GR;
    while (read >> press >> tempra >> Td >> RH >> Precp >> hour >> GR)
    {
        write << press <<" "<< tempra<<" "<<Td<<" "<<RH<<" "<<hour<<" "<<GR<<" "<<Precp<<endl;
    }

    read.close();
    write.close();
    
}

int main(){
    string name , num;
    cout << "input the name and num : " << endl;
    cin >> name >> num;

    write(name , num);
    return 0;
}