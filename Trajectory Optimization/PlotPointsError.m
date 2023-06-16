function PlotPointsError(k, coordinates,list_eta1,list_puav,D_k,size_square,No,wc,T)
    axes('Xlim', [0 size_square+5],'Ylim', [0 size_square+5], 'XTick', 0:5:size_square, 'YTick', 0:5:size_square);
    title('Number of data and eta for No. Users ='+string(No))
    for itr = 1:T
        subplot(2,2,itr)
        plot(wc(1),wc(2),'blueD','MarkerSize',5)
        hold on
        plot(list_puav(itr,1),list_puav(itr,2),'redD','MarkerSize',5)
        text(list_puav(itr,1),list_puav(itr,2),string(itr),'VerticalAlignment','bottom','HorizontalAlignment','center','color','black');
        hold on
        for s=1:k
            plot(coordinates(s,1),coordinates(s,2), 'green.','MarkerSize',10)
            t1 = strcat('(',string(D_k(s)));
            t2= strcat(t1,')');
            t4 = strcat(t2,'&(');
            eta1 = list_eta1(itr,:);
            t5= strcat(t4,string(eta1(s)));
            t6= strcat(t5,')');
            text(coordinates(s,1),coordinates(s,2),t6,'VerticalAlignment','bottom','HorizontalAlignment','center','color','black');
        end
    end
   
    
    
   
end