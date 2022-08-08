#colo_list = [
'price',
'sqft_living',
'grade',
'sqft_above',
'sqft_living15',
'bathrooms', 
'view',
'sqft_basement',
'bedrooms',
'lat',
'zipcode_98004',    
'waterfront',      
'floors',           
'zipcode_98039',    
'zipcode_98040',   
'zipcode_98112',    
'zipcode_98006',    
'yr_renovated',     
'zipcode_98033',    
'zipcode_98105',    
'sqft_lot',         
'zipcode_98075',    
'zipcode_98199',    
'sqft_lot15',
'zipcode_98168',   
'zipcode_98001',   
'zipcode_98042',  
'zipcode_98023',
'long'         
#]




X = df_modelisation[['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view','sqft_basement',
'bedrooms','zipcode_98004','waterfront','floors','zipcode_98039','zipcode_98040','zipcode_98112','zipcode_98006',
'yr_renovated','zipcode_98033','zipcode_98105','sqft_lot','zipcode_98075','zipcode_98199','sqft_lot15','zipcode_98001',
'zipcode_98042','zipcode_98023']]
y = df_modelisation[['price']]