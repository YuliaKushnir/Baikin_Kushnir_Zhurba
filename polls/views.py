from django.db.models import F
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import generic

from .models import Choice, Question
from .comments_analyzer import analyze_sentiment

class IndexView(generic.ListView):
    template_name = "polls/index.html"
    context_object_name = "latest_question_list"

    def get_queryset(self):
        return Question.objects.order_by("-pub_date")[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = "polls/detail.html"


class ResultsView(generic.DetailView):
    model = Question
    template_name = "polls/results.html"


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST["choice"])
    except (KeyError, Choice.DoesNotExist):
        return render(
            request,
            "polls/detail.html",
            {
                "question": question,
                "error_message": "Ви не вибрали варіант",
            },
        )
    else:
        selected_choice.votes = F("votes") + 1
        comment = request.POST.get("comment", "")
        selected_choice.comment = comment
        sentiment = analyze_sentiment(comment)  # Аналіз настрою
        print(f"Настрій коментаря: {sentiment}")  # Вивід результату для налагодження
        selected_choice.save()
        return HttpResponseRedirect(reverse("polls:results", args=(question.id,)))
