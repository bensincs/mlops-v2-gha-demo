projectName="benmlops"
roleName="Contributor"
subscriptionId="fde39ef3-ba50-49cd-821b-d929985bc10f"
environment="Dev" #First letter should be capitalized
servicePrincipalName="Azure-ARM-${environment}-${projectName}"
# Verify the ID of the active subscription
echo "Using subscription ID $subscriptionId"
echo "Creating SP for RBAC with name $servicePrincipalName, with role $roleName and in scopes     /subscriptions/$subscriptionId"
az ad sp create-for-rbac --name $servicePrincipalName --role $roleName --scopes /subscriptions/$subscriptionId --json-auth
echo "Please ensure that the information created here is properly save for future use."