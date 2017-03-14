<?php
if (!array_key_exists("password",$_POST) or $_POST["password"] != "qVK0fFt6zKLH{6T")
	die("Access denied");

if (!array_key_exists("sql",$_POST))
	die("No query is recieved");

$db = new PDO("sqlite:AFchallenge");
$statement = $db->prepare($_POST["sql"]);
$statement->execute();
if (!$statement)
    $data['error'] = $db->errorInfo();
else {
	$data['error'] = 'OK';
	$data['data']  = $statement->fetchAll(PDO::FETCH_ASSOC);
}

echo json_encode($data);
?>