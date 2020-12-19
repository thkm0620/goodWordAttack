#include <cstdio>
#include <cstring>

int main(){
    FILE *fp;
    FILE *fp2;
    fp=fopen("uci.txt","r"); // uci repository file
    fp2=fopen("trainData.txt","w"); // format must be changed to .txt -> .xlsx(trainData) , -> .csv(testdata)
                                    // When changing format to .xlsx, set ^ as a mark to change cell
    int startNum=1000, endNum=5000; // (0,1000) to make testdata.csv
    char msg[500]={0,};
    //fprintf(fp2,"COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS\n"); // (only for testdata) csv file format in preceding code

    for(int i=0; i<=endNum; i++){
        int check=1; // 1 : spam 0: ham
        fscanf(fp,"%s",msg);
        if(msg[0]=='h') check=0;
        fscanf(fp,"%c",&msg[0]);
        fscanf(fp,"%[^\n]",msg);
        //for(int j=0; j<sizeof(msg); j++) if(msg[j]==',') msg[j]='.'; // (only for testdata) to make csv file
        if(i>startNum){
            if(check || i%5==0)
                fprintf(fp2,"%s^%d\n",msg,check);
            // spam -> print, ham-> print at a probability of 1/5 (trainData)
            // fprint ID,AUTHOR,2020-12-11, (testdata) csv file format in preceding code
        }
        memset(msg,0,sizeof(msg));
    }
    fclose(fp);
    fclose(fp2);
}
