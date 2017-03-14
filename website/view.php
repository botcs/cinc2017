<html>
<head>

<title>View human classification result | PPKE ITK - AF Challenge</title>

<meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1'>
<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />
	
<link rel='shortcut icon' href='../core/favicon.ico' type='image/x-icon' />

<!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">

<!-- jQuery library -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>

<!-- Latest compiled JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>

<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<script type="text/javascript">
google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawChart);

$(document).ready(function() {
	$('.btn').click(function() {
		getData($(this).attr("data-ID"));
	});
});

function getData(rec_ID) {
	$.post(
		"query.php",
		{ password: "qVK0fFt6zKLH{6T", sql: "SELECT data FROM recordings WHERE rec_ID = " + rec_ID },
		function(json, status){
			var data = JSON.parse(json);
			var values = data["data"][0].data.split(',');
			xy = [];
			for (var i = 0; i < values.length; i++) {
				xy.push([i/300, Number(values[i])]);
			}
			drawChart(rec_ID, xy);
		}
	);
}

function drawChart(rec_ID, xy) {
	var data = new google.visualization.DataTable();
	data.addColumn('number', 'X');
	data.addColumn('number', 'Y');

	data.addRows(xy);

	var options = {
	width: 700,
	height: 300,
	hAxis: {
	  title: 'Time [s]'
	},
	explorer: {
		maxZoomIn: 0.125,
		maxZoomOut: 2,
		keepInBounds: true,
		actions: ['dragToZoom', 'rightClickToReset']
	},'tooltip' : {
	  trigger: 'none'
	},
	legend: {position: 'none'},
	chartArea: {left:0,top:0,width:'80%',height:'80%'},
	enableInteractivity: false
	};

	var chart = new google.visualization.LineChart(document.getElementById(rec_ID));
	chart.draw(data, options);
}
</script>
</head>

<body>
<p style="text-align: center; font-weight: 600; width: 90%; margin:auto">Zoom in: selecting an area on the chart, zoom out: right click.</p>
<div id="container" style="width: 90%; margin:auto; position: relative; top: 50px;"><?php

$dict = array( "N" => "Normal", "A" => "Atrial Fibrillation", "O" => "Other", "~" => "Noisy");
$db = new PDO("sqlite:AFchallenge");
$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
$result = $db->prepare("
	SELECT rec_ID, name, source, class, machine_guess, human_guess, comment_before, comment_after
	FROM recordings
	WHERE human_guess IS NOT NULL
");
$result->execute();
while($feedback = $result->fetch(PDO::FETCH_ASSOC)) {?>
	<div class="row">
		<div class="col-sm-4">
		<table class="table table-striped table-bordered"><tbody>
			<tr><td>Name:</td><td><?php echo $feedback['name'];?></td></tr>
			<!--<tr><td>Source:</td><td><?php echo $feedback['source'];?></td></tr>-->
			<tr><td>Class:</td><td><?php echo $feedback['class'];?></td></tr>
			<tr><td>Machine guess:</td><td><?php echo $feedback['machine_guess'];?></td></tr>
			<tr><td>Human guess:</td><td><?php echo $feedback['human_guess'];?></td></tr>
			<tr><td>Comment before:</td><td><?php echo $feedback['comment_before'];?></td></tr>
			<tr><td>Comment after:</td><td><?php echo $feedback['comment_after'];?></td></tr>
		</tbody></table>
		</div>
		<div id="<?php echo $feedback['rec_ID']; ?>" class="col-sm-8">
			<div style="transform: translate(50%, 350%);">
				<button name="load" class="btn btn-default" data-ID="<?php echo $feedback['rec_ID']; ?>">Load chart</button>
			</div>
		</div>
	</div>
<?php
}?>
</div>
</body>
</html>





































