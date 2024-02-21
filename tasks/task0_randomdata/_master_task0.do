* Project: Multi-country B2B Networks ECB.
* Author: Glenn Magerman (glenn.magerman@ulb.be)
* First version: Feb 2020.
* Current version: Feb 2024.


// change to task directory
clear all
cd "$task0"

// initialize task (build from inputs)
foreach dir in tmp output {
	cap !rm -rf "`dir'"
}

// create task folders
foreach dir in input src output tmp {
	cap !mkdir "`dir'"
}	
	
// code	
	do "src/1_create_randomdata.do" 
	
// maintenance
cap !rm -rf "tmp"									// Unix
cap !rmdir /q /s "tmp"								// Windows		

// back to main folder of tasks
cd "$folder"		
