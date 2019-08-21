$(function() {
    $('button#process').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_input_helper', {
        question_data: $('#question_data').val()
      }, function(data) {
        $("#result").prepend(data.result);
      });
      return false;
    });
  });

$(function() {
    $('button#store-btn').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_store_context', {
        text_data: $('#text_data').val(),
      }, function(data) {
          if (data.context) {
              $("#store-context").hide();
              $("#text_data").val("");
              $("#reset-context").show();
              $("#question-input").show();
              $("#context-title").html(data.title)
              $("#context-data").html(data.context)
          }
      });
      return false;
    });
  });

$('button#reset-btn').on('click', function() {
    $("#store-context").show();
    $("#reset-context").hide();
    $("#question-input").hide();
    $(".history").show();
    $("#history").prepend($("#result").html());
    $("#history").prepend(("<p>" + $("#context-data").html()  + "</p>").replace("hilite", ""));
    $("#history").prepend(("<h4>" + $("#context-title").html() + "</h4>").replace("hilite", ""));
    $("#context-title").html("");
    $("#context-data").html("");
    $("#result").html("");
});

$(function() {
    $('button#random-btn').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_random_page', {
      }, function(data) {
        $("#text_data").val(data.context);
      });
      return false;
    });
});

function highlight(element, start, end) {
    if (start > -1) {
        var item = $(element);
        var str = item.data("origHTML");
        if (!str) {
            str = item.html();
            item.data("origHTML", str);
        }
        str = str.substr(0, start) +
            '<span class="hilite">' +
            str.substr(start, end - start + 1) +
            '</span>' +
            str.substr(end + 1);
        item.html(str);
    }
}

function restore(element, start, end) {
    if (start > -1) {
        var item = $(element);
        var str = item.data("origHTML");
        if (!str) {
            str = item.html();
            item.data("origHTML", str);
        }
        str = str.substr(0, start) +
            '<span class="hilite">' +
            str.substr(start, end - start + 1) +
            '</span>' +
            str.substr(end + 1);
        item.html(str);
    }
}
