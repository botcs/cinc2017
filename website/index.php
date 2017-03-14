<html>
<head>

<title>Human Classification | PPKE ITK - AF Challenge</title>

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

function drawChart() {
	var data = new google.visualization.DataTable();
	data.addColumn('number', 'X');
	data.addColumn('number', 'Y');

	data.addRows([<?php
	$db = new PDO("sqlite:AFchallenge");
	$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
	$result = $db->prepare("
		SELECT count(*)
		FROM recordings
		WHERE human_guess IS NULL AND machine_GUESS IS NOT NULL AND machine_guess <> class
	");
	$result->execute();
	$countOfMachineGuesses = $result->fetch(PDO::FETCH_NUM);
	
	if ($countOfMachineGuesses[0] == 0)
		$result = $db->prepare("
			SELECT rec_ID
			FROM recordings
			WHERE human_guess IS NULL AND comment_before IS NULL AND comment_after IS NULL
		");
	else
		$result = $db->prepare("
			SELECT rec_ID
			FROM recordings
			WHERE human_guess IS NULL AND machine_GUESS IS NOT NULL AND machine_guess <> class
		");
	$result->execute();
	$IDs = $result->fetchALL(PDO::FETCH_NUM);
	$index = rand(0, count($IDs)-1);
	$result = $db->prepare("SELECT data, class FROM recordings WHERE rec_ID = ?");
	$result->execute(array($IDs[$index][0]));
	$row = $result->fetch(PDO::FETCH_ASSOC);
	$data = explode(',',$row['data']);
	$length = count($data);
	$str = "";

	for ($i = 1; $i < $length; $i += 3){
		$str .= "[" . $i/300. . ", $data[$i]]" . ($i < $length-1 ? ", " : "");	
	}
	echo $str;
	?>]);

	var options = {
	width: 1200,
	height: 400,
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
	chartArea: {left:100,top:50,width:'80%',height:'75%'},
	enableInteractivity: false
	};

	var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
	chart.draw(data, options);
}
</script>
</head>

<body><?php

if (!empty($_POST)) {
	$db = new PDO("sqlite:AFchallenge");
	$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
	$result = $db->prepare("UPDATE recordings SET comment_after = ? WHERE rec_ID = ?");
	$result->execute(array($_POST['comment'], $_POST['ID']));
}

?>

<div id="chart_div" data-ID="<?php echo $IDs[$index][0]; ?>" data-class="<?php echo $row['class']; ?>" style="width: 100%; height: 400px"></div>
<p style="text-align: center; font-weight: 600; width: 90%; margin:auto">Zoom in: selecting an area on the chart, zoom out: right click.</p>

<div id="form-container" style="width: 90%; margin:auto">
	<form id="start-form" class="form-horizontal">
	<fieldset>

	<!-- Multiple Radios (inline) -->
	<div class="form-group">
	  <label class="col-sm-4 control-label" for="radios">Class</label>
	  <div class="col-sm-5"> 
		<label class="radio-inline" for="radios-0">
		  <input type="radio" name="radios" id="radios-0" value="1" checked="checked">
		  Normal
		</label> 
		<label class="radio-inline" for="radios-1">
		  <input type="radio" name="radios" id="radios-1" value="2">
		  Atrial Fibrillation
		</label> 
		<label class="radio-inline" for="radios-2">
		  <input type="radio" name="radios" id="radios-2" value="3">
		  Other
		</label> 
		<label class="radio-inline" for="radios-3">
		  <input type="radio" name="radios" id="radios-3" value="4">
		  Noisy
		</label> 
		<label class="radio-inline" for="radios-4">
		  <input type="radio" name="radios" id="radios-4" value="5">
		  Not sure
		</label>
	  </div>
	</div>

	<!-- Textarea -->
	<div class="form-group">
	  <label class="col-sm-4 control-label" for="comment">Comment</label>
	  <div class="col-sm-5">                     
		<textarea class="form-control" id="comment" name="comment"></textarea>
	  </div>
	</div>

	<!-- Button -->
	<div class="form-group">
	<label class="col-sm-4 control-label" for="submit"></label>
	  <div class="col-sm-5">
		<button id="submit" name="submit" class="btn btn-default">Submit</button>
	  </div>
	</div>

	</fieldset>
	</form>
</div>

<script>
$(document).ready(function() {
	$('#submit').click(function() {
	$.post(
		"save.php",
		{ ID: $("#chart_div").attr("data-ID"),
		  real_class: $("#chart_div").attr("data-class"),
		  radios: $('input[name=radios]:checked').val(),
		  comment: $("#comment").val() },
		function(data, status){ $('#form-container').html(data); }
	);
	});
	$("#start-form").submit(function(e) {
		e.preventDefault();
	});
});
</script>

</body>
</html>