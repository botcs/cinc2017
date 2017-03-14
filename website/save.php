<?php

$dict = array( 1 => "N", 2 => "A", 3 => "O", 4 => "~", 5 => "?");
$dict2 = array( "N" => "Normal", "A" => "Atrial Fibrillation", "O" => "Other", "~" => "Noisy", "?" =>"Not sure");
$db = new PDO("sqlite:AFchallenge");
$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
$result = $db->prepare("UPDATE recordings SET human_guess = ?, comment_before = ? WHERE rec_ID = ?");
$result->execute(array($dict[$_POST['radios']], $_POST['comment'], $_POST['ID']));
?>

<form id="save-form" class="form-horizontal" action="./" method="post">
<fieldset>

<!-- Form Name -->
<legend>Class of the recording: <?php echo $dict2[$_POST['real_class']]; ?> (your guess was: <?php echo $dict2[$dict[$_POST['radios']]]; ?>)</legend>

<textarea style="display:none" id="ID" name="ID"><?php echo $_POST['ID']; ?></textarea>
	
<!-- Textarea -->
<div class="form-group">
  <label class="col-md-4 control-label" for="comment">Further comments</label>
  <div class="col-md-4">                     
    <textarea class="form-control" id="comment" name="comment"></textarea>
  </div>
</div>

<!-- Button -->
<div class="form-group">
  <label class="col-md-4 control-label" for="next"></label>
  <div class="col-md-4">
    <button id="next" name="next" class="btn btn-default">Next</button>
  </div>
</div>

</fieldset>
</form>
