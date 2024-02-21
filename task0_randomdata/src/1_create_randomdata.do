* Project: Multi-country B2B Networks ECB.
* Author: Glenn Magerman (glenn.magerman@ulb.be)
* First version: Feb 2020.
* Current version: Feb 2024.

/*______________________________________________________________________________

This file generates random datasets that have the same variable names as in the 
real data used for this project.
In particular:
- BE_network_2002_2022: year, vat_i (seller), vat_j (buyer), value (value i to j).
- 

Please change the first 2 letters of the name of the dataset to reflect the 
2-letter ISO code of your country.

The number of firms and linke can be changed at will to test and debug codes.
Small numbers speed up all calculations. Larger numbers might be needed with 
multiple cells (e.g. at least 5 firms in a zip-code-sector group). 
The seed ensures random draws are identical across draws. 
(But might vary across user Operating Systems or software used).

______________________________________________________________________________*/

*------------
**# 0. Macros 
*------------ 
// first and last years of panel data 
global start  = 2002  
global end    = 2022
global start_1 = $start + 1

// number of firms and number of links 
global nfirms = 1000
global nlinks = 10000

// set seed for reproducibility 
set seed 8718354

*---------------------
* 1. Firm-to-firm data
*---------------------
forvalues t = $start/$end {
	clear
	set obs $nlinks
	gen year = `t'
	gen vat_i = floor(($nfirms - 1)*runiform() + 1) 		// create random links
	gen vat_j = floor(($nfirms - 1)*runiform() + 1)
	drop if vat_i==vat_j 									// drop potential self-loops
	duplicates drop vat_i vat_j, force 						// don't allow for multiple edges
	gen sales_ij = 250+ exp(rnormal()*5 + 5)  				// link values drawn from log-normal distribution
	save "tmp/network_`t'", replace
}	

// collect all cross-sections and create panel 
use "tmp/network_$start", clear
	forvalues t = $start_1/$end {
		append using  "tmp/network_`t'"
	}
save  "output/BE_network_${start}_${end}", replace	

*--------------------------------------
* 2. annual accounts + VAT declarations
*--------------------------------------
forvalues t = $start/$end {
	clear
	set obs $nfirms	
	gen year = `t'
	gen vat = _n											
	
// generate balanced panel (for unbalanced, draw from uniform distribution)	
	foreach x in turnover inputs_total laborcost {			// firm variables 
		gen `x' = floor(exp(rnormal()*5) + 1) + 1000
	}
	foreach x in fte {
		gen `x' = floor(exp(rnormal()*2) + 1)				// employment in FTEs
	}	
save "./tmp/firms_`t'", replace						
}

use "./tmp/firms_$start", clear
	forvalues t = 2003/2014 {
		append using  "./tmp/firms_`t'"
	}	
save  "./output/BE_annac_${start}_${end}", replace	
	

*-----------------------
* 3. NACE + postal codes
*-----------------------
forvalues t = $start/$end {
	clear
	set obs $nfirms	
	gen year = `t'
	
// generate balanced panel (for unbalanced, draw from uniform distribution)	
	gen vat = _n											
	
// NACE sectors 	
	gen nace = 100 + int((9600)*runiform()) 				// 4-digit NACE
    replace nace = floor(nace/10)*10	                    // scale back on # sectors to have enough firms within each sector
	
// zip codes 	
	gen zip = 100*(10 + int((99-10+1)*runiform())) 	// zip code
save "tmp/nace_zip_`t'", replace						
}

use "tmp/nace_zip_$start", clear
	forvalues t = 2003/2014 {
		append using  "tmp/nace_zip_`t'"
	}
save  "output/BE_nace_zip_${start}_${end}", replace	

clear
