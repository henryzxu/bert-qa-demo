$(function() {
    $('button#process').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_input_helper', {
        question_data: $('#question-data').val()
      }, function(data) {
        $("#result").prepend(data.result);
        var hilite = new Function(data.highlight_script);
        hilite()
      });
      return false;
    });
  });

$('textarea').each(function () {
  this.setAttribute('style', 'height:' + (this.scrollHeight) + 'px;overflow-y:hidden;');
}).on('input', function () {
  this.style.height = 'auto';
  this.style.height = (this.scrollHeight) + 'px';
});

$(function() {
    $('button#store-btn').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_store_context', {
        text_data: $('#text-data').val(),
      }, function(data) {
          if (data.context) {
              $("#store-context").hide();
              $("#text-data").val("");
              $("#reset-context").show();
              $("#question-input").show();
              $("#question-data").val("");
              $('textarea').trigger('input');
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

    var hist = $("#history");
    hist.prepend("<br>");
    hist.prepend($("#result").html());
    hist.prepend($("#context-data").html().replace('<span class="hilite">', "").replace('</span>', ""));
    hist.prepend(("<h4>" + $("#context-title").html() + "</h4>").replace('<span class="hilite">', "").replace('</span>', ""));

    $("#context-title").html("");
    $("#context-data").html("");
    $("#result").html("");
    $('textarea').trigger('input');
});

$('a#toggle-history').on('click', function() {
    var hist_toggle = $('#toggle-history');
    hist_toggle.text() === "(hide)" ? hist_toggle.text("(show)") : hist_toggle.text("(hide)");
    $("#history").toggle();
});

$(function() {
    $('button#random-btn').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_random_page', {
      }, function(data) {
        $("#text-data").val(data.context);
        $('textarea').trigger('input');
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
